# Databricks notebook source
import os
from pyspark.sql.functions import col, to_date

# Widgets for parameters
dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "default")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

print(f"Ingesting NYC Taxi data to catalog={catalog}, schema={schema}...")

# Create schema if not exists
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")

# Read from standard Databricks NYC taxi sample table
trips_df = spark.table("samples.nyctaxi.trips")

# Clean/Select columns for regression task (predicting fare_amount)
# We limit to 50,000 records to keep execution fast and light
processed_df = (
    trips_df
    .filter(col("fare_amount") > 0)
    .filter(col("trip_distance") > 0)
    .select(
        col("tolls_amount"),
        col("trip_distance"),
        col("fare_amount"),
        col("pickup_datetime"),
        col("dropoff_datetime"),
        col("pickup_zip"),
        col("dropoff_zip"),
        col("payment_type")
    )
    .limit(50000)
)

# Split into reference (baseline training data) and current (live production features)
# The dataset contains trips from 2016. We split around mid-February.
reference_df = processed_df.filter(col("pickup_datetime") < "2016-02-15 00:00:00")
current_df = processed_df.filter(col("pickup_datetime") >= "2016-02-15 00:00:00").drop("fare_amount") # production features have no label initially

# Write reference table to Unity Catalog
ref_table = f"{catalog}.{schema}.nyctaxi_reference"
reference_df.write.format("delta").mode("overwrite").saveAsTable(ref_table)
print(f"✓ Reference table saved to {ref_table} ({reference_df.count()} rows)")

# Write current table to Unity Catalog
cur_table = f"{catalog}.{schema}.nyctaxi_current"
current_df.write.format("delta").mode("overwrite").saveAsTable(cur_table)
print(f"✓ Current table saved to {cur_table} ({current_df.count()} rows)")
