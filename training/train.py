import os
import logging
import numpy as np
import pandas as pd
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── Config ─────────────────────────────────────────────────────────────────
#
# All config comes from environment variables — never hardcoded.
# Docker Compose injects these from your .env file at startup.
#
MLFLOW_URI       = os.getenv("MLFLOW_TRACKING_URI",    "http://mlflow:5000")
EXPERIMENT       = os.getenv("MLFLOW_EXPERIMENT_NAME", "fraud-detection")
DATABASE_URL     = os.getenv("DATABASE_URL",           "postgresql://admin:admin@postgres:5432/fraud_db")
MODEL_NAME       = os.getenv("MODEL_NAME",             "fraud-detection-model")


# These are the columns model learns from.
# They must exist in both your training data AND the API request body.
# If you add a feature here, you must add it everywhere.
#
FEATURES = [
    "amount",
    "hour_of_day",
    "day_of_week",
    "merchant_category",
    "distance_from_home",
    "transaction_count",
    "is_foreign",
]


# XGBoost hyperparameters.
# Keeping them in one dict makes it easy to log them all to MLflow
# with a single mlflow.log_params(PARAMS) call.
#
# scale_pos_weight handles class imbalance — in real fraud data,
# maybe 1 in 100 transactions is fraud. Without this, the model
# learns to just predict "not fraud" every time and gets 99% accuracy
# while being completely useless.
# Rule of thumb: set it to (number of legitimate / number of fraud).
#
PARAMS = {
    "n_estimators":     200,    # number of trees
    "max_depth":        6,      # how deep each tree grows
    "learning_rate":    0.1,    # how much each tree corrects the previous
    "subsample":        0.8,    # use 80% of rows per tree (reduces overfitting)
    "colsample_bytree": 0.8,    # use 80% of features per tree
    "scale_pos_weight": 10,     # compensates for fraud being rare
    "eval_metric":      "logloss",
    "random_state":     42,
}


# ── Data loading ───────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """
    Try to load real data from Postgres first.
    Fall back to synthetic data if the table is empty.

    In production you'd always have real data. The synthetic fallback
    means this project works out of the box without a dataset.
    """
    try:
        engine = create_engine(DATABASE_URL)
        with engine.connect() as conn:
            # Check if transactions table has any rows
            count = conn.execute(text("SELECT COUNT(*) FROM transactions")).scalar()

        if count > 0:
            df = pd.read_sql("SELECT * FROM transactions", engine)
            logger.info(f"Loaded {len(df):,} rows from Postgres.")
            return df

    except Exception as e:
        logger.warning(f"Could not load from Postgres: {e}")

    logger.warning("No data in Postgres — generating synthetic dataset.")
    return generate_synthetic()


def generate_synthetic(n: int = 10_000) -> pd.DataFrame:
    """
    Generate realistic-looking fraud data for demonstration.

    Key design decisions:
    - 2% fraud rate (realistic for credit cards)
    - Fraudulent transactions tend to be larger, happen at night,
      and come from foreign merchants — these patterns let XGBoost
      actually learn something meaningful.
    """
    rng = np.random.default_rng(42)  # fixed seed = reproducible results

    n_fraud = int(n * 0.02)   # 200 fraud cases
    n_legit = n - n_fraud     # 9800 legitimate

    # Legitimate transactions — normal patterns
    legit = pd.DataFrame({
        "amount":             rng.lognormal(3.5, 1.2, n_legit),   # ~$33 median
        "hour_of_day":        rng.integers(6, 23,  n_legit),      # daytime
        "day_of_week":        rng.integers(0, 7,   n_legit),
        "merchant_category":  rng.integers(0, 20,  n_legit),
        "distance_from_home": rng.exponential(20,  n_legit),      # close to home
        "transaction_count":  rng.integers(1, 10,  n_legit),
        "is_foreign":         rng.integers(0, 2,   n_legit),
        "is_fraud":           np.zeros(n_legit, int),
    })

    # Fraudulent transactions — suspicious patterns
    fraud = pd.DataFrame({
        "amount":             rng.lognormal(5.0, 1.5, n_fraud),   # ~$148 median, higher variance
        "hour_of_day":        rng.integers(0, 5,   n_fraud),      # late night
        "day_of_week":        rng.integers(0, 7,   n_fraud),
        "merchant_category":  rng.integers(0, 20,  n_fraud),
        "distance_from_home": rng.exponential(200, n_fraud),      # far from home
        "transaction_count":  rng.integers(5, 30,  n_fraud),      # many transactions
        "is_foreign":         rng.choice([0, 1], n_fraud, p=[0.2, 0.8]),  # mostly foreign
        "is_fraud":           np.ones(n_fraud, int),
    })

    # Combine and shuffle so fraud isn't all at the end
    df = pd.concat([legit, fraud], ignore_index=True).sample(frac=1, random_state=42)
    logger.info(f"Generated {len(df):,} synthetic rows ({n_fraud} fraud, {n_legit} legit).")
    return df


# ── Training ───────────────────────────────────────────────────────────────

def train():
    df = load_data()

    if "is_fraud" not in df.columns:
        raise ValueError("Dataset must contain an 'is_fraud' column.")

    X = df[FEATURES]
    y = df["is_fraud"].astype(int)

    #
    # stratify=y ensures both train and test sets have the same
    # fraud ratio. Without this, with 2% fraud you might get
    # all frauds in training and none in test, or vice versa.
    #
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    logger.info(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")

    # ── MLflow run ─────────────────────────────────────────────────────────
    #
    # Everything inside `with mlflow.start_run()` is tracked.
    # MLflow creates a unique run ID, timestamps it, and links all
    # params/metrics/artifacts to that run.
    #
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run():

        # Train
        model = XGBClassifier(**PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # Evaluate
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]  # probability of fraud

        metrics = {
            "accuracy":  accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall":    recall_score(y_test, y_pred, zero_division=0),
            "f1":        f1_score(y_test, y_pred, zero_division=0),
            "roc_auc":   roc_auc_score(y_test, y_prob),
        }

        for name, value in metrics.items():
            logger.info(f"  {name:12s}: {value:.4f}")

        #
        # Log params and metrics to MLflow.
        # These show up in the MLflow UI at http://localhost:5000
        # so you can compare runs side by side.
        #
        mlflow.log_params(PARAMS)
        mlflow.log_metrics(metrics)

        #
        # Log (save) the model to MLflow.
        # MLflow serialises the XGBoost model and uploads it to MinIO
        # as an S3 artifact. registered_model_name gives it a human
        # name so the API can load it with:
        #   mlflow.xgboost.load_model("models:/fraud-detection-model/latest")
        # instead of needing the raw run ID.
        #
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )

        run_id = mlflow.active_run().info.run_id
        logger.info(f"✅ Training complete. Run ID: {run_id}")
        logger.info(f"   Model registered as: {MODEL_NAME}")
        logger.info(f"   Artifacts stored in: MinIO (s3://mlflow-artifacts/)")


if __name__ == "__main__":
    train()