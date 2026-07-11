# Databricks MLOps Pipeline (NYC Taxi Dataset)

This directory contains a Databricks Asset Bundle (DAB) to deploy and run the MLOps pipeline natively on a Databricks workspace. It uses Unity Catalog for database tables and Model Registry, Serverless Workflows compute for low-cost execution, and Databricks Lakehouse Monitoring to track prediction and features drift.

---

## Directory Structure
* **`databricks.yml`:** The bundle configuration file defining the jobs, environments, variables, and the `quality_monitors` resource.
* **`notebooks/`:**
  * **`01_ingest.py`:** Reads the built-in `samples.nyctaxi.trips` dataset and creates the reference and current/live features tables.
  * **`03_train.py`:** Trains a regression model to predict trip fare amounts and registers it to Unity Catalog Model Registry with the `challenger` alias.
  * **`04_evaluate.py`:** Evaluates the challenger against the active champion model, promoting the challenger if its RMSE is lower.
  * **`05_inference.py`:** Loads the `@champion` model, performs batch prediction on current live features, and appends outputs to the inference log table.
  * **`06_monitor.py`:** Uses the Databricks Python SDK to trigger a refresh on the deployed Lakehouse Monitor.

---

## Serverless Compute
The bundle is configured to run all Job tasks on **Databricks Serverless Workflows Compute** via the task `environment_key` setting. This avoids the overhead and cold-start latency of provisioning virtual machines (job clusters) manually.

---

## Data & Model Monitoring
Monitoring is defined declaratively as a **Lakehouse Quality Monitor** resource inside `databricks.yml`:
- It is deployed directly onto the `nyctaxi_inference` table.
- A baseline reference table (`nyctaxi_reference`) is specified for comparison.
- It is configured as an **Inference Log Profile** for a regression problem, measuring feature drift and model accuracy (RMSE/MAE over time) as actual labels arrive.

---

## Local Deployment & Testing

To deploy and run the bundle manually from your local machine, ensure the Databricks CLI is installed and configured:

1. **Validate the bundle configuration:**
   ```bash
   databricks bundle validate
   ```
2. **Deploy the bundle resources to Databricks:**
   ```bash
   databricks bundle deploy
   ```
3. **Trigger the deployed MLOps job:**
   ```bash
   databricks bundle run taxi_mlops_job
   ```

---

## CI/CD Pipeline (GitHub Actions)
The repository includes a GitHub Action workflow inside `.github/workflows/deploy.yml` that validates and deploys your Databricks Asset Bundle automatically upon pushes to the `databricks/**` directory.

### Authentication Config:
The workflow uses secure **Service Principal OAuth**. You must configure the following Secrets in your GitHub repository:
- `DATABRICKS_HOST`: Your Databricks workspace URL.
- `DATABRICKS_CLIENT_ID`: The Application/Client ID of your Service Principal.
- `DATABRICKS_CLIENT_SECRET`: The Client Secret of your Service Principal.
