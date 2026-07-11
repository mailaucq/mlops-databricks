# Databricks notebook source
import mlflow
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np

# Widgets for parameters
dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "default")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

model_name = f"{catalog}.{schema}.nyctaxi_model"

print(f"Evaluating NYC Taxi models. Registry model: {model_name}...")

# Configure MLflow to use Unity Catalog for the Model Registry
mlflow.set_registry_uri("databricks-uc")
client = mlflow.MlflowClient()

# Load evaluation data
ref_table = f"{catalog}.{schema}.nyctaxi_reference"
df = spark.table(ref_table).toPandas()

NUMERIC = ["trip_distance"]
CATEGORICAL = ["pickup_zip", "dropoff_zip"]
target_col = "fare_amount"

X_val = df[NUMERIC + CATEGORICAL]
y_val = df[target_col]

# Load challenger model
try:
    challenger_ver = client.get_model_version_by_alias(model_name, "challenger").version
    challenger_model = mlflow.sklearn.load_model(f"models:/{model_name}@challenger")
except Exception as e:
    print(f"Error: Challenger model not found: {e}")
    raise e

# Load champion model
try:
    champion_ver = client.get_model_version_by_alias(model_name, "champion").version
    champion_model = mlflow.sklearn.load_model(f"models:/{model_name}@champion")
except Exception:
    champion_model = None
    champion_ver = None

# Evaluate challenger
y_pred_chal = challenger_model.predict(X_val)
rmse_chal = np.sqrt(mean_squared_error(y_val, y_pred_chal))
print(f"Challenger (v{challenger_ver}) RMSE: {rmse_chal:.4f}")

# Evaluate champion (if exists) and compare
promote = False
if champion_model is None:
    print("No existing champion model. Challenger will be promoted directly.")
    promote = True
else:
    y_pred_champ = champion_model.predict(X_val)
    rmse_champ = np.sqrt(mean_squared_error(y_val, y_pred_champ))
    print(f"Champion (v{champion_ver}) RMSE: {rmse_champ:.4f}")
    
    # Promote if challenger RMSE is lower (better)
    delta = rmse_champ - rmse_chal
    print(f"RMSE Improvement (delta): {delta:.4f}")
    promote = delta > 0.01

# Promote challenger to champion
if promote:
    client.set_registered_model_alias(model_name, "champion", challenger_ver)
    print(f"✓ Version {challenger_ver} promoted to Champion!")
else:
    print("Challenger did not outperform Champion. Champion remains unchanged.")
