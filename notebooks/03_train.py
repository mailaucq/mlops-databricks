# Databricks notebook source
import mlflow
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from mlflow.models import infer_signature
import numpy as np

# Widgets for parameters
dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "default")
dbutils.widgets.text("experiment_name", "/Shared/taxi_mlops_experiment")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
experiment_name = dbutils.widgets.get("experiment_name")

model_name = f"{catalog}.{schema}.nyctaxi_model"

print(f"Training NYC Taxi model. Registering to: {model_name}...")

# Configure MLflow to use Unity Catalog for the Model Registry
mlflow.set_registry_uri("databricks-uc")

try:
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        mlflow.create_experiment(name=experiment_name)
except Exception as e:
    print(f"Skipped programmatically creating experiment: {e}")

mlflow.set_experiment(experiment_name)

# Load reference dataset
ref_table = f"{catalog}.{schema}.nyctaxi_reference"
df = spark.table(ref_table).toPandas()

# Preprocess features
NUMERIC = ["trip_distance"]
CATEGORICAL = ["pickup_zip", "dropoff_zip"]
target_col = "fare_amount"

X = df[NUMERIC + CATEGORICAL]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ("prep", ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)
    ])),
    ("model", RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1))
])

# Track with MLflow
with mlflow.start_run(run_name="train_taxi_model") as run:
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    print(f"Evaluation Metrics: RMSE={rmse:.4f}, R2={r2:.4f}")
    
    signature = infer_signature(X_test.head(5), y_pred[:5])
    mlflow.sklearn.log_model(pipeline, artifact_path="model", signature=signature)
    
# Register model to Unity Catalog
ver = mlflow.register_model(model_uri=f"runs:/{run.info.run_id}/model", name=model_name)

# Set challenger alias
client = mlflow.MlflowClient()
client.set_registered_model_alias(model_name, "challenger", ver.version)
print(f"✓ Registered model version {ver.version} as challenger.")
