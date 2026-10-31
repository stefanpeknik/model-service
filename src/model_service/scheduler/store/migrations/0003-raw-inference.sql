-- An execution can run one raw model request instead of a builder. Its request
-- and response bytes live in inference_payloads, outside the frequently scanned
-- executions rows.
-- depends: 0002-builders-and-models

ALTER TABLE executions ADD COLUMN operation TEXT NOT NULL DEFAULT 'builder'
    CHECK (operation IN ('builder', 'inference'));
ALTER TABLE executions ALTER COLUMN builder_id DROP NOT NULL;
ALTER TABLE executions ADD CHECK ((operation = 'builder') = (builder_id IS NOT NULL));

CREATE TABLE inference_payloads (
    execution_id UUID PRIMARY KEY REFERENCES executions (id) ON DELETE CASCADE,
    request BYTEA,
    response BYTEA
);

-- Keep maintenance cheap when many finished executions are retained.
CREATE INDEX executions_deadline_idx ON executions (deadline_at)
    WHERE work_kind IS NOT NULL;
CREATE INDEX executions_report_deadline_idx ON executions (terminal_report_deadline_at)
    WHERE work_kind IS NOT NULL;
CREATE INDEX executions_lease_idx ON executions (lease_expires_at)
    WHERE fencing_token IS NOT NULL;
CREATE INDEX executions_finished_idx ON executions (updated_at)
    WHERE work_kind IS NULL;

DROP INDEX executions_active_processing_idx;
CREATE INDEX executions_active_processing_idx
    ON executions (model_id, operation, lease_expires_at)
    WHERE work_kind = 'processing' AND fencing_token IS NOT NULL;
