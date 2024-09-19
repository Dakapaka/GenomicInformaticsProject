import anndata as ad
import scanpy as sc
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

adata = ad.read_h5ad("Mouse_brain_cell_bin.h5ad")

print(adata.n_obs)
print(adata.n_vars)

if 'spatial' not in adata.obsm.keys():
    raise ValueError("The 'spatial' coordinates are missing from the dataset.")
if 'annotation' not in adata.obs.keys():
    raise ValueError("The 'annotation' field is missing from the dataset.")

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

spot_size = 30.0

original_annotation_plot_name = 'spatial_original_annotations_brain.png'
try:
    if 'annotation_colors' not in adata.uns:
        sc.pl.spatial(adata, color='annotation', title="Original Annotations Brain", spot_size=spot_size, save=original_annotation_plot_name)
        existing_palette = adata.uns['annotation_colors']  # Extract and save the color palette
    else:
        sc.pl.spatial(adata, color='annotation', title="Original Annotations Brain", spot_size=spot_size, save=original_annotation_plot_name)
        existing_palette = adata.uns['annotation_colors']  # Use the existing color palette if present
except Exception as e:
    print(f"Error while saving the original annotations plot: {e}")

# Perform PCA for dimensionality reduction
sc.tl.pca(adata, svd_solver='arpack')
pca_result = adata.obsm['X_pca'][:, :8]  # Top 8 PCA components

# Combine PCA with spatial coordinates
spatial_coords = adata.obsm['spatial']
if spatial_coords.shape[0] != pca_result.shape[0]:
    raise ValueError("Mismatch between PCA results and spatial coordinates dimensions.")
X = np.hstack([pca_result, spatial_coords])

# Extract and encode cell annotations
y = adata.obs['annotation'].to_numpy()
if np.any(pd.isnull(y)):  # Check for missing values using pandas
    raise ValueError("The annotation data contains missing values.")
label_encoder = LabelEncoder()
y_numeric = label_encoder.fit_transform(y)

# Print the original annotation distribution
unique_orig, counts_orig = np.unique(y, return_counts=True)
print(f"Original annotation distribution: {dict(zip(unique_orig, counts_orig))}")

def plot_barplot_with_counts(x_labels, y_values, title, file_name):
    plt.figure(figsize=(max(12, len(x_labels) * 0.5), 6))
    ax = sns.barplot(x=x_labels, y=y_values)
    plt.xticks(rotation=90)
    plt.title(title)
    plt.ylabel('Count')
    for i, v in enumerate(y_values):
        ax.text(i, v + 0.02 * max(y_values), str(v), color='black', ha='center')
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()

# Plot original annotation distribution
plot_barplot_with_counts(unique_orig, counts_orig, 'Original Annotation Distribution', 'original_annotation_distribution.png')

# Define the specific combinations of K and L values to use
K_L_combinations = [(3, 2), (5, 3), (7, 4)]

# Compute class weights to handle class imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(y_numeric), y=y_numeric)
class_weight_dict = dict(zip(np.unique(y_numeric), class_weights))

# Define classifiers to test
classifiers = {
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, 
                             scale_pos_weight=1)  # Set scale_pos_weight to 1 for multiclass
}

# Function to plot confusion matrix with adjustable figure size
def plot_confusion_matrix(fold, clf_name, K, L, y_true, y_pred, class_names, title):
    cm = confusion_matrix(y_true, y_pred)
    num_classes = len(class_names)
    fig_width = max(10, num_classes * 0.5)
    fig_height = fig_width * 0.75
    plt.figure(figsize=(fig_width, fig_height))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=90, ax=plt.gca())
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_fold{fold}_{clf_name}_K{K}_L{L}_brain.png')
    plt.close()

# Plot percentage of explained variance by PCA components
explained_variance_ratio = adata.uns['pca']['variance_ratio']
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio * 100, alpha=0.7, color='b')
plt.title('Percentage of Explained Variance by PCA Components')
plt.xlabel('Number of PCA Components')
plt.ylabel('Percentage of Explained Variance')
plt.grid(True)
plt.savefig('figures/pca_explained_variance_percentage_brain.png')
plt.close()

# Function to plot feature importance
def plot_feature_importance(model, feature_names, clf_name, K, L, top_n=20):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[-top_n:]  # Top N features
        plt.figure(figsize=(10, 6))
        plt.title(f"Feature Importance (Top {top_n}) - {clf_name} K={K} L={L}")
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel("Relative Importance")
        plt.tight_layout()
        plt.savefig(f'figures/feature_importance_{clf_name}_K{K}_L{L}.png')
        plt.close()

# Feature names (PCA components and spatial coordinates)
num_pca_features = pca_result.shape[1]
num_spatial_features = spatial_coords.shape[1]
feature_names = [f'PC{i+1}' for i in range(num_pca_features)] + [f'Spatial_{i+1}' for i in range(num_spatial_features)]

# Iterate over K and L combinations
for K, L in K_L_combinations:
    # Initialize Stratified K-Fold cross-validator
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)

    # Iterate over classifiers
    for clf_name, classifier in classifiers.items():
        print(f"\nTesting classifier: {clf_name} with K={K}, L={L}")

        # Initialize arrays for storing metrics
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []

        # Initialize a placeholder for refined annotations
        refined_annotations = adata.obs['annotation'].copy()
        all_predictions = np.zeros((adata.n_obs, K), dtype=int)

        # Train and evaluate the classifier using K-fold cross-validation
        for fold, (train_index, test_index) in enumerate(skf.split(X, y_numeric)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y_numeric[train_index], y_numeric[test_index]

            # Apply feature scaling
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Train the model
            classifier.fit(X_train, y_train)

            # Make predictions on both train and test sets
            y_pred_train = classifier.predict(X_train)
            y_pred_test = classifier.predict(X_test)

            # Store predictions for both train and test sets
            all_predictions[train_index, fold] = y_pred_train
            all_predictions[test_index, fold] = y_pred_test

            # Calculate metrics on the test set
            accuracy = accuracy_score(y_test, y_pred_test)
            precision = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)

            # Collect scores
            accuracy_scores.append(accuracy)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

            # Plot confusion matrix for the current fold
            plot_confusion_matrix(fold, clf_name, K, L, y_test, y_pred_test, class_names=label_encoder.classes_, title=f'Confusion Matrix - Fold {fold + 1} ({clf_name}) K = {K}, L = {L}')

            # Print fold metrics
            print(f"Fold {fold + 1} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

        # Calculate and print average metrics for this classifier
        print(f"\nMean Accuracy for {clf_name} with K={K}, L={L}: {np.mean(accuracy_scores):.4f}")
        print(f"Mean Precision: {np.mean(precision_scores):.4f}")
        print(f"Mean Recall: {np.mean(recall_scores):.4f}")
        print(f"Mean F1-score: {np.mean(f1_scores):.4f}")

        # Refinement of annotations based on majority and threshold
        changed_annotations_count = 0
        for i in range(adata.n_obs):
            pred_count = Counter(all_predictions[i])
            most_common_pred, count = pred_count.most_common(1)[0]

            if count > L:
                refined_numeric_annotation = most_common_pred
            else:
                refined_numeric_annotation = y_numeric[i]  # Keep the original annotation if not refined

            refined_annotations[i] = label_encoder.inverse_transform([refined_numeric_annotation])[0]

            # Count changes in annotation
            if refined_numeric_annotation != y_numeric[i]:
                changed_annotations_count += 1

        # Print unique refined annotations
        unique_refined, counts_refined = np.unique(refined_annotations, return_counts=True)
        print(f"Unique refined annotations for {clf_name} with K={K}, L={L}: {dict(zip(unique_refined, counts_refined))}")
        print(f"Total annotations changed for {clf_name} with K={K}, L={L}: {changed_annotations_count}")

        # Plot distribution of refined annotations
        plot_barplot_with_counts(unique_refined, counts_refined, f'Refined Annotation Distribution ({clf_name}, K={K}, L={L})', f'refined_annotation_distribution_{clf_name}_K{K}_L{L}.png')

        # Save the refined annotations into AnnData object
        adata.obs[f'refined_annotation_{clf_name}_K{K}_L{L}'] = refined_annotations

        # Set the existing palette for consistent coloring
        adata.uns[f'refined_annotation_{clf_name}_K{K}_L{L}_colors'] = existing_palette

        # Save the plot of the refined annotations with consistent coloring
        refined_annotation_plot_name = f'spatial_refined_annotations_{clf_name}_K{K}_L{L}_brain.png'
        try:
            sc.pl.spatial(adata, color=f'refined_annotation_{clf_name}_K{K}_L{L}', 
                          title=f"Refined Annotations (K={K}, L={L}, {clf_name}) Brain", 
                          spot_size=spot_size, save=refined_annotation_plot_name)
        except Exception as e:
            print(f"Error while saving the refined annotations plot for {clf_name} with K={K} and L={L}: {e}")

        # Plot feature importance for classifiers that support it
        plot_feature_importance(classifier, feature_names, clf_name, K, L, top_n=20)
