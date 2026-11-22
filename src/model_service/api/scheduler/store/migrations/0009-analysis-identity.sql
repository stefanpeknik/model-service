-- An execution runs an analysis: its ID, the served models it calls and, when
-- admitted as an analysis, the resolved document every attempt sends and its
-- digest. Builder runs map to the day-one analysis of their builder and model;
-- other pairs keep `<builder>:<model>` as a legacy ID. Their model is their one
-- served model. Submission keys and fingerprints are unchanged.
-- depends: 0008-invocation-configuration

ALTER TABLE executions
    ADD COLUMN analysis_id TEXT,
    ADD COLUMN models TEXT[],
    ADD COLUMN analysis_digest TEXT,
    ADD COLUMN analysis JSONB,
    ADD CHECK ((analysis IS NULL) = (analysis_digest IS NULL));

UPDATE executions SET
    analysis_id = CASE
        WHEN builder_id = 'heatmap' AND model_id = 'prostate-classifier-1'
            THEN 'prostate-heatmap'
        WHEN builder_id = 'heatmap' AND model_id = 'episeg-1'
            THEN 'episeg-heatmap'
        WHEN builder_id = 'tile-embeddings' AND model_id = 'virchow2'
            THEN 'virchow2-embeddings'
        WHEN builder_id = 'tile-embeddings' AND model_id = 'prov-gigapath'
            THEN 'prov-gigapath-embeddings'
        ELSE builder_id || ':' || model_id
    END,
    models = ARRAY[model_id];

ALTER TABLE executions
    ALTER COLUMN analysis_id SET NOT NULL,
    ALTER COLUMN models SET NOT NULL;

DROP INDEX IF EXISTS executions_active_processing_idx;
ALTER TABLE executions DROP COLUMN builder_id, DROP COLUMN model_id;

CREATE INDEX executions_active_processing_idx ON executions (lease_expires_at)
    WHERE work_kind = 'processing' AND fencing_token IS NOT NULL;
