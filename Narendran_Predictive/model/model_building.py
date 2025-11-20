# ===============================================
# Predictive Maintenance Model Training Pipeline
# ===============================================
import os
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import xgboost as xgb
from huggingface_hub import HfApi
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

# ---- Constants ----
HF_TOKEN = os.getenv("HF_TOKEN")
api_client = HfApi(token=HF_TOKEN)

DATA_DIR = "/content/Narendran_Predictive/model"
os.makedirs(DATA_DIR, exist_ok=True)

HF_DATA_REPO = "Narendranh/narendran_predictive_data"
HF_MODEL_REPO = "Narendranh/Narendran_PredictiveMaintenance-XGBoost-Model"

FILE_PATHS = {
    "X_train": os.path.join(DATA_DIR, "Xtrain.csv"),
    "X_test": os.path.join(DATA_DIR, "Xtest.csv"),
    "y_train": os.path.join(DATA_DIR, "ytrain.csv"),
    "y_test": os.path.join(DATA_DIR, "ytest.csv"),
    "model": os.path.join(DATA_DIR, "best_model.pkl")
}

EXPERIMENT_NAME = "Engine_Narendran_Predictive"
mlflow.set_experiment(EXPERIMENT_NAME)


# =========================================
# Helper Functions
# =========================================
def load_data():
    """Load training and test data."""
    try:
        X_train = pd.read_csv(FILE_PATHS["X_train"])
        X_test = pd.read_csv(FILE_PATHS["X_test"])
        y_train = pd.read_csv(FILE_PATHS["y_train"]).iloc[:, 0]
        y_test = pd.read_csv(FILE_PATHS["y_test"]).iloc[:, 0]
        print("[INFO] Data successfully loaded from local splits.")
        return X_train, X_test, y_train, y_test
    except Exception as err:
        print(f"[FATAL] Unable to load dataset splits. Error: {err}")
        raise SystemExit(1)


def build_pipeline():
    """Create a preprocessing + XGBoost pipeline."""
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('xgb_model', xgb.XGBClassifier(
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        ))
    ])
    return model_pipeline


def tune_model(pipeline, X_train, y_train):
    """Perform hyperparameter tuning using GridSearchCV."""
    param_grid = {
        'xgb_model__n_estimators': [100, 200],
        'xgb_model__max_depth': [3, 5],
        'xgb_model__learning_rate': [0.01, 0.1]
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='f1',
        cv=3,
        verbose=1,
        n_jobs=-1
    )

    print("[INFO] Starting GridSearchCV for XGBoost tuning...")
    grid_search.fit(X_train, y_train)
    print("[INFO] Hyperparameter tuning completed successfully.")
    return grid_search.best_estimator_, grid_search.best_params_


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance on the test set."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }

    print("\n--- Model Evaluation Results ---")
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.4f}")

    return metrics


def save_and_upload_model(model):
    """Save the model locally and upload to Hugging Face."""
    joblib.dump(model, FILE_PATHS["model"])
    print(f"[INFO] Model saved locally at: {FILE_PATHS['model']}")

    try:
        api_client.create_repo(repo_id=HF_MODEL_REPO, repo_type="model", exist_ok=True)
        api_client.upload_file(
            path_or_fileobj=FILE_PATHS["model"],
            path_in_repo=os.path.basename(FILE_PATHS["model"]),
            repo_id=HF_MODEL_REPO,
            repo_type="model"
        )
        print(f"[SUCCESS] Model uploaded to Hugging Face Hub: {HF_MODEL_REPO}")
    except Exception as err:
        print(f"[ERROR] Model upload to Hugging Face failed. Details: {err}")


# =========================================
# Main Workflow
# =========================================
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    pipeline = build_pipeline()

    print("\n--- Initiating MLflow Run for XGBoost Model Training ---")
    with mlflow.start_run(run_name="XGBoost_GridSearch_Tuning") as run:
        best_model, best_params = tune_model(pipeline, X_train, y_train)

        # Log best parameters
        mlflow.log_params(best_params)
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("scoring_metric", "f1")

        # Evaluate and log metrics
        metrics = evaluate_model(best_model, X_test, y_test)
        for key, value in metrics.items():
            mlflow.log_metric(f"test_{key}", value)

        # Log the model artifact to MLflow
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model_artifact",
            registered_model_name="XGBoostPredictiveMaintenance"
        )
        print("[INFO] Model logged to MLflow successfully.")

    # Save locally and upload to Hugging Face
    save_and_upload_model(best_model)
