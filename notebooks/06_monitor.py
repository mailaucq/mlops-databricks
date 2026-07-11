# Databricks notebook source
from databricks.sdk import WorkspaceClient

# Widgets for parameters
dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "default")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

from databricks.sdk.service.catalog import MonitorInferenceLog, MonitorInferenceLogProblemType

table_name = f"{catalog}.{schema}.nyctaxi_inference"

print(f"Checking/Refreshing Quality Monitor for table: {table_name}...")

w = WorkspaceClient()

try:
    print("Attempting to create Quality Monitor for table...")
    w.lakehouse_monitors.create(
        table_name=table_name,
        inference_log=MonitorInferenceLog(
            problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_REGRESSION,
            model_id_col="model_id",
            prediction_col="prediction",
            label_col="fare_amount",
            timestamp_col="pickup_datetime",
            granularities=["1 day"]
        ),
        output_schema_name=f"{catalog}.{schema}",
        assets_dir="/Shared/taxi_mlops_monitoring"
    )
    print("✓ Quality Monitor successfully created!")
except Exception as e:
    # If the monitor already exists or creation fails, we print the log and trigger a refresh
    print(f"Monitor creation skipped (probably already exists). Detail: {e}")
    print("Triggering monitor refresh...")
    refresh_info = w.lakehouse_monitors.run_refresh(table_name=table_name)
    print(f"✓ Lakehouse Monitor refresh successfully triggered! Refresh ID: {refresh_info.refresh_id}")
