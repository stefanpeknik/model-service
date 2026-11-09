-- An execution can run one raw model request instead of a builder. Its request
-- and response bytes are files on the inference storage volume; the row keeps
-- their relative paths and sizes.
-- depends: 0001-create-executions

ALTER TABLE executions
    ADD COLUMN operation TEXT NOT NULL DEFAULT 'builder'
        CHECK (operation IN ('builder', 'inference')),
    ALTER COLUMN builder_id DROP NOT NULL,
    ADD CHECK ((operation = 'builder') = (builder_id IS NOT NULL)),
    ADD COLUMN request_path TEXT,
    ADD COLUMN request_size BIGINT,
    ADD COLUMN response_path TEXT,
    ADD COLUMN response_size BIGINT,
    ADD CHECK ((request_path IS NULL) = (request_size IS NULL)),
    ADD CHECK ((response_path IS NULL) = (response_size IS NULL)),
    ADD CHECK (operation = 'inference' OR (request_path IS NULL AND response_path IS NULL));

DROP INDEX executions_active_processing_idx;
CREATE INDEX executions_active_processing_idx
    ON executions (model_id, operation, lease_expires_at)
    WHERE work_kind = 'processing' AND fencing_token IS NOT NULL;
