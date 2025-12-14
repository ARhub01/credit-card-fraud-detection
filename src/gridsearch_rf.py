from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from pipeline import build_pipeline

X_train, X_test, y_train, y_test = build_pipeline('data/creditcard.csv')

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20]
}

rf = RandomForestClassifier(class_weight='balanced', random_state=42)

grid = GridSearchCV(
    rf, param_grid, scoring='roc_auc', cv=3, n_jobs=-1
)

grid.fit(X_train, y_train)

print('Best Parameters:', grid.best_params_)
print('Best ROC-AUC:', grid.best_score_)