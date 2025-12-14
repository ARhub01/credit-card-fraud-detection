from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve
)
import matplotlib.pyplot as plt
import os


def evaluate(model, X_test, y_test, model_name=None):
    """
    Evaluate a trained classification model on test data.
    Saves ROC and Precision-Recall curves.
    """

    # Create reports directory if it doesn't exist
    os.makedirs("reports", exist_ok=True)

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Text metrics
    print("\n" + "=" * 50)
    if model_name:
        print(f"Evaluation Results: {model_name}")
    print(classification_report(y_test, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

    # ---------- ROC Curve ----------
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    roc_path = f"reports/roc_curve_{model_name}.png" if model_name else "reports/roc_curve.png"
    plt.savefig(roc_path)
    plt.close()

    # ---------- Precision-Recall Curve ----------
    precision, recall, _ = precision_recall_curve(y_test, y_prob)

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    pr_path = f"reports/pr_curve_{model_name}.png" if model_name else "reports/pr_curve.png"
    plt.savefig(pr_path)
    plt.close()
