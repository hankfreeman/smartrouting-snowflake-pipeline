-- =============================================================================
-- Persist scoring logs from the SPCS event table into a permanent SCORING_LOG
-- table. Replace <YOUR_DATABASE>, <YOUR_SCHEMA>, <EVENT_DATABASE>, <EVENT_SCHEMA>
-- with your account identifiers before running.
--
-- Before running: replace <YOUR_WAREHOUSE> with your warehouse name.
-- =============================================================================

-- 1. Create the permanent scoring log table (run once)
CREATE TABLE IF NOT EXISTS <YOUR_DATABASE>.<YOUR_SCHEMA>.SCORING_LOG (
    input_phone     VARCHAR,
    input_source    VARCHAR,
    model           VARCHAR,
    log_timestamp   VARCHAR,
    score           FLOAT,
    tier            VARCHAR,
    raw_response    VARCHAR,
    event_ts        TIMESTAMP_LTZ,
    inserted_at     TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP()
);

-- 2. Task: copy new log rows from the account event table into SCORING_LOG
CREATE OR REPLACE TASK <YOUR_DATABASE>.<YOUR_SCHEMA>.SYNC_SCORING_LOG_FROM_EVENTS
    WAREHOUSE = <YOUR_WAREHOUSE>
    SCHEDULE = '5 MINUTE'
AS
INSERT INTO <YOUR_DATABASE>.<YOUR_SCHEMA>.SCORING_LOG (
    input_phone,
    input_source,
    model,
    log_timestamp,
    score,
    tier,
    raw_response,
    event_ts
)
SELECT
    TRY_PARSE_JSON(e.VALUE):input_phone::VARCHAR,
    TRY_PARSE_JSON(e.VALUE):input_source::VARCHAR,
    TRY_PARSE_JSON(e.VALUE):model::VARCHAR,
    TRY_PARSE_JSON(e.VALUE):timestamp::VARCHAR,
    TRY_PARSE_JSON(e.VALUE):score::FLOAT,
    TRY_PARSE_JSON(e.VALUE):tier::VARCHAR,
    TRY_PARSE_JSON(e.VALUE):raw_response::VARCHAR,
    e.TIMESTAMP
FROM <EVENT_DATABASE>.<EVENT_SCHEMA>.SERVICE_LOGS e
WHERE e.RECORD_TYPE = 'LOG'
  AND e.VALUE IS NOT NULL
  AND TRY_PARSE_JSON(e.VALUE):input_phone IS NOT NULL
  AND (e.RESOURCE_ATTRIBUTES:snow.service.name::VARCHAR = 'SMART_ROUTING_5TOWER_SERVICE'
       OR e.RESOURCE_ATTRIBUTES:snow.service.name IS NULL)
  AND e.TIMESTAMP > COALESCE(
        (SELECT MAX(event_ts) FROM <YOUR_DATABASE>.<YOUR_SCHEMA>.SCORING_LOG),
        '1970-01-01'::TIMESTAMP_LTZ
      );

ALTER TASK <YOUR_DATABASE>.<YOUR_SCHEMA>.SYNC_SCORING_LOG_FROM_EVENTS RESUME;

-- Query historical logs:
--   SELECT * FROM <YOUR_DATABASE>.<YOUR_SCHEMA>.SCORING_LOG ORDER BY event_ts DESC LIMIT 100;
