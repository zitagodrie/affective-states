# Libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import spearmanr 
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, PredictionErrorDisplay, log_loss


def regression_evaluation(y_test, y_pred):
    # Calculate MSE, RMSE, MAE,  Spearmann corr. , Pearson corr., concordance correlation coefficient (ccc)
    mse = np.mean((y_test - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test - y_pred))
    spearman_corr, _ = spearmanr(y_test, y_pred)
    pearson_corr = np.corrcoef(y_test, y_pred)[0, 1]
    ccc = (2 * spearman_corr) / (1 + spearman_corr)
    return mse, rmse, mae, spearman_corr, pearson_corr, ccc

def test_regression_evaluation():
    # Example usage:
    y_test = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 2.1, 2.9, 4.2, 5.1])
    
    mse, rmse, mae, spearman_corr, pearson_corr, ccc = regression_evaluation(y_test, y_pred)
    print(f"MSE: {mse}, RMSE: {rmse}, MAE: {mae}, Spearman Correlation: {spearman_corr}, Pearson Correlation: {pearson_corr}, CCC: {ccc}")

def plot_classification_report(report, title='Classification Report', cmap='Blues'):
    """
    Plot a classification report as a heatmap with precision, recall, F1-score, and support.
    """
    # Convert report to DataFrame if it's not already
    if isinstance(report, dict):
        report_df = pd.DataFrame(report).T
    else:
        report_df = report.copy()
    
    # Ensure columns are in the correct order
    metrics = ['precision', 'recall', 'f1-score', 'support']
    report_df = report_df[metrics]
    
    # Normalize support column for better visualization
    report_df['support'] = report_df['support'] / report_df['support'].max()
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        report_df.iloc[:-1, :-1],  # Exclude the last row (accuracy) and support column
        annot=True,
        fmt=".2f",
        cmap=cmap,
        cbar_kws={'label': 'Score'},
        linewidths=0.5,
        ax=ax
    )
    
    # Customize plot
    ax.set_title(title, pad=20)
    ax.set_xlabel("Metrics")
    ax.set_ylabel("Classes")
    plt.tight_layout()
    
    return fig, ax

def test_plot_classification_report():
    # Example usage:
    report = {
        'tableware': {'precision': 0.000, 'recall': 0.000, 'f1-score': 0.000, 'support': 3},
        'container': {'precision': 0.000, 'recall': 0.000, 'f1-score': 0.000, 'support': 2},
        'veh_float': {'precision': 0.462, 'recall': 1.000, 'f1-score': 0.632, 'support': 6},
        'headlamps': {'precision': 0.000, 'recall': 0.000, 'f1-score': 0.000, 'support': 3},
        'tid_float': {'precision': 0.933, 'recall': 0.933, 'f1-score': 0.933, 'support': 15},
        'tid_non': {'precision': 0.933, 'recall': 1.000, 'f1-score': 0.966, 'support': 14},
    }

    plot_classification_report(report, title='DecisionTreeClassifier Classification Report')
    return plt.show()


def plot_class_prediction_error(cm, class_names=None, title="Class Prediction Error"):
    """
    Plot a class prediction error chart showing actual vs. predicted class distributions.
    
    Args:
        cm (array-like): Confusion matrix or 2D array of shape (n_classes, n_classes).
        class_names (list): Names of classes (e.g., ['tableware', 'container', ...]).
        title (str): Plot title.
    """

    # Normalize by row (actual class) to show proportions
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create DataFrame for plotting
    if class_names is None:
        class_names = [f"Class {i}" for i in range(cm.shape[0])]
    df_cm = pd.DataFrame(cm_normalized, index=class_names, columns=class_names)
    
    # Plot stacked bars with switched axes
    fig, ax = plt.subplots(figsize=(10, 6))
    df_cm.T.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
    
    # Customize plot
    ax.set_title(title, pad=20)
    ax.set_xlabel("Actual Class")
    ax.set_ylabel("Proportion of Predictions")
    ax.legend(title="Predicted Class", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    return fig, ax

def test_plot_class_prediction_error():
    # Example usage:
    cm = np.array([[5, 0, 0], [0, 3, 2], [0, 1, 4]])
    class_names = ['Class A', 'Class B', 'Class C']
    
    plot_class_prediction_error(cm, class_names=class_names)
    return plt.show()


def multi_classification_evaluation(y_true, y_pred):
    # Multiclass Classification report (precision, recall, f1, support), Multiclass confusion matrix, Class Prediction Error plot
    report = classification_report(y_true, y_pred, output_dict=True)
    fig1, ax = plot_classification_report(report)
    # fig1.savefig(os.path.join(BASE_DIR, "./figures", "classification_report.png"))
    fig1.show()

    cm = confusion_matrix(y_true, y_pred)
    cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig2 = cm_disp.plot(cmap='Blues')
    # fig2.savefig(os.path.join(BASE_DIR, "./figures", "confusion_matrix.png"))
    plt.show()

    fig3, ax3 = plot_class_prediction_error(cm, class_names=cm_disp.display_labels)
    # fig3.savefig(os.path.join(BASE_DIR, "./figures", "prediction_error.png"))
    fig3.show()

    return fig1, fig2, fig3

def test_multi_classification_evaluation():
    # Example usage:
    y_true = np.array([0, 1, 2, 2, 0, 1])
    y_pred = np.array([0, 1, 2, 0, 0, 1])

    multi_classification_evaluation(y_true, y_pred)
    return plt.show()

if __name__ == "__main__":
    # Test functions
    test_regression_evaluation()
    # test_plot_classification_report()
    # test_plot_class_prediction_error()
    test_multi_classification_evaluation()