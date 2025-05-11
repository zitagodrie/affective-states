# Libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, log_loss
)

def regression_evaluation(y_test, y_pred):
    # Calculate MSE, RMSE, MAE, Spearman corr., Pearson corr., concordance correlation coefficient (CCC)
    mse = np.mean((y_test - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test - y_pred))
    spearman_corr, _ = spearmanr(y_test, y_pred)
    pearson_corr = np.corrcoef(y_test, y_pred)[0, 1]
    ccc = (2 * spearman_corr) / (1 + spearman_corr)
    return mse, rmse, mae, spearman_corr, pearson_corr, ccc

def test_regression_evaluation():
    y_test = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 2.1, 2.9, 4.2, 5.1])
    results = regression_evaluation(y_test, y_pred)
    print(f"MSE: {results[0]}, RMSE: {results[1]}, MAE: {results[2]}, "
          f"Spearman Correlation: {results[3]}, Pearson Correlation: {results[4]}, CCC: {results[5]}")

def plot_classification_report(report, classes, title='Classification Report ', cmap='Blues'):
    if isinstance(report, dict):
        report_df = pd.DataFrame(report).T
    else:
        report_df = report.copy()

    metrics = ['precision', 'recall', 'f1-score', 'support']
    report_df = report_df[metrics]
    report_df = report_df.reindex(classes)

    report_df['support'] = report_df['support'].astype(int)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        report_df,  # Exclude the normalized support column
        annot=True,
        fmt=".2f",
        cmap=cmap,
        cbar_kws={'label': 'Score'},
        linewidths=0.5,
        ax=ax,
        vmin=0,  # Set the minimum value of the color scale
        vmax=1   # Set the maximum value of the color scale
    )
    ax.set_title(title, pad=20)
    ax.set_xlabel("Metrics")
    ax.set_ylabel("Classes")
    plt.tight_layout()
    return fig, ax

def test_plot_classification_report():
    report = {
        'tableware': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 3},
        'container': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 2},
        'veh_float': {'precision': 0.462, 'recall': 1.0, 'f1-score': 0.632, 'support': 6},
        'headlamps': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 3},
        'tid_float': {'precision': 0.933, 'recall': 0.933, 'f1-score': 0.933, 'support': 15},
        'tid_non': {'precision': 0.933, 'recall': 1.0, 'f1-score': 0.966, 'support': 14},
    }
    plot_classification_report(report, classes=list(report.keys()), title='DecisionTreeClassifier Classification Report')
    plt.show()

def plot_class_prediction_error(cm, classes, title="Class Prediction Error"):
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    df_cm = pd.DataFrame(cm_normalized, index=classes, columns=classes)

    fig, ax = plt.subplots(figsize=(10, 6))
    df_cm.T.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
    ax.set_title(title, pad=20)
    ax.set_xlabel("Actual Class")
    ax.set_ylabel("Proportion of Predictions")
    ax.legend(title="Predicted Class", labels=df_cm.columns, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    return fig, ax

def test_plot_class_prediction_error():
    y_true = np.array(["low", "high", "neutral", "low", "low", "high"])
    y_pred = np.array(["neutral", "high", "low", "low", "high", "high"])
    cm = confusion_matrix(y_true, y_pred)
    plot_class_prediction_error(cm)
    plt.show()

def multi_classification_evaluation(y_true, y_pred, subtitle="", display=True):
    classes = ['very low', 'low', 'neutral', 'high', 'very high']
    classes = [cls for cls in classes if cls in y_true]

    # Multiclass Classification report (precision, recall, f1, support), Multiclass confusion matrix, Class Prediction Error plot
    report = classification_report(y_true, y_pred, output_dict=True)
    fig1, ax = plot_classification_report(report, classes, title="Classification report " + subtitle)
    if display:
        fig1.show()

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(report.keys())[:-3])
    fig2 = cm_disp.plot(cmap='Blues')
    cm_disp.ax_.set_title("Confusion Matrix " + subtitle)
    if display:
        plt.show()

    fig3, ax3 = plot_class_prediction_error(cm, classes, title="Class Prediction Error " + subtitle)
    if display:
        fig3.show()

    statistics =  {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred, average='weighted'),
        "classification_report": report
    }
    print(f"Accuracy: {statistics['accuracy']}")
    print(f"F1 Score: {statistics['f1_score']}")
    print(f"Classification Report: {statistics['classification_report']}")

    return fig1, fig2, fig3, statistics

def test_multi_classification_evaluation():
    y_true = np.array(["low", "high", "neutral", "low", "low", "high"])
    y_pred = np.array(["neutral", "high", "low", "low", "high", "high"])
    multi_classification_evaluation(y_true, y_pred, "subtitle")
    plt.show()

if __name__ == "__main__":
    test_regression_evaluation()
    test_multi_classification_evaluation()