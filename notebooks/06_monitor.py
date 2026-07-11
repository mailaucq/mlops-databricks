# Databricks notebook source
from databricks.sdk import WorkspaceClient

# Widgets for parameters
dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "default")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

table_name = f"{catalog}.{schema}.nyctaxi_inference"

print(f"Triggering Databricks Lakehouse Monitoring refresh for table: {table_name}...")

# Initialize Databricks Workspace Client
# The client automatically authenticates using the active notebook session credentials
w = WorkspaceClient()

try:
    # Trigger refresh on the table monitor
    refresh_info = w.lakehouse_monitors.run_refresh(table_name=table_name)
    print(f"✓ Lakehouse Monitor refresh successfully triggered! Refresh ID: {refresh_info.refresh_id}")
except Exception as e:
    print(f"⚠ Warning: Could not trigger monitor refresh. Please ensure the Quality Monitor "
          f"has been deployed via Databricks Asset Bundles first.\nError detail: {e}")
