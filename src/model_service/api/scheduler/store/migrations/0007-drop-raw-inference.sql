-- Raw inference no longer runs as an execution: tiles go straight to Ray Serve.
-- Remaining inference executions are removed with the operation column; dropping
-- a column also drops the CHECK constraints and index that use it. Builder rows
-- are kept. The request and response files on the old inference volume are not
-- touched; delete that volume once no older scheduler uses it.
-- depends: 0006-priority-scale

DELETE FROM executions WHERE operation = 'inference';

ALTER TABLE executions
    DROP COLUMN request_path,
    DROP COLUMN request_size,
    DROP COLUMN response_path,
    DROP COLUMN response_size,
    DROP COLUMN operation;

ALTER TABLE executions ALTER COLUMN builder_id SET NOT NULL;

DROP INDEX IF EXISTS executions_active_processing_idx;
CREATE INDEX executions_active_processing_idx ON executions (model_id, lease_expires_at)
    WHERE work_kind = 'processing' AND fencing_token IS NOT NULL;
