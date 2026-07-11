# Databricks notebook source
import mlflow
import pandas as pd

# Widgets for parameters
dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "default")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

model_name = f"{catalog}.{schema}.nyctaxi_model"

print(f"Performing batch inference. Registry model: {model_name}...")

# Configure MLflow to use Unity Catalog for the Model Registry
mlflow.set_registry_uri("databricks-uc")

# Load champion model
try:
    champion_model = mlflow.sklearn.load_model(f"models:/{model_name}@champion")
    print("✓ Successfully loaded champion model.")
except Exception as e:
    print(f"Error: Champion model not found: {e}")
    raise e

# Load current production features
cur_table = f"{catalog}.{schema}.nyctaxi_current"
current_df = spark.table(cur_table)

if current_df.count() == 0:
    print("No production features found to perform inference on.")
else:
    # Convert features to Pandas for inference
    df_pandas = current_df.toPandas()
    
    NUMERIC = ["trip_distance"]
    CATEGORICAL = ["pickup_zip", "dropoff_zip"]
    X = df_pandas[NUMERIC + CATEGORICAL]
    
    # Run batch predictions
    predictions = champion_model.predict(X)
    df_pandas["prediction"] = predictions
    df_pandas["model_id"] = "nyctaxi_model"
    
    # We simulate ground truth (actual fare_amount) arriving after the ride.
    # We add minor random noise to mock real labels for performance monitoring.
    import numpy as np
    rng = np.random.default_rng(42)
    df_pandas["fare_amount"] = (df_pandas["prediction"] + rng.normal(0, 1.5, len(df_pandas))).clip(2.5, 500.0)
    
    # Convert back to Spark DataFrame
    spark_pred_df = spark.createDataFrame(df_pandas)
    
    # Append predictions to the monitored inference log table
    inference_table = f"{catalog}.{schema}.nyctaxi_inference"
    spark_pred_df.write.format("delta").mode("append").saveAsTable(inference_table)
    print(f"✓ Appended {spark_pred_df.count()} predictions to inference table: {inference_table}")
