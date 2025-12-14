from src.pipeline import build_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from src.evaluate import evaluate
import joblib

DATA_PATH = "data/creditcard.csv"

X_train, X_test, y_train, y_test = build_pipeline(DATA_PATH)

models = {
    "logistic": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42
    ),
    "xgboost": XGBClassifier(
        scale_pos_weight=10, eval_metric="logloss", random_state=42
    )
}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    joblib.dump(model, f"models/{name}.pkl")

    evaluate(model, X_test, y_test, model_name=name)

print("\n✅ All models trained and evaluated successfully")
