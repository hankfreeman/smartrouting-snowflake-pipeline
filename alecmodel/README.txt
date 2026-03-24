Place Alec metadata here for train_alec_model_replica.py:
  close_rate_model_v4_metadata.pkl

Without this file, the Alec replica step will not run. You can still train the base 5-tower model
and deploy with deploy_5tower_snowflake.py (non-Alec Dockerfile).
