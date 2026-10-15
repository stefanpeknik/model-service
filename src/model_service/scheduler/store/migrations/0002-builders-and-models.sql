-- Executions run a builder with a selected model instead of a per-model workflow.
-- depends: 0001-create-executions

ALTER TABLE executions RENAME COLUMN workflow_id TO builder_id;
ALTER TABLE executions ADD COLUMN model_id TEXT;

UPDATE executions SET
    model_id = CASE builder_id
        WHEN 'prostate-heatmap' THEN 'prostate-classifier-1'
        WHEN 'episeg-segmentation' THEN 'episeg-1'
        WHEN 'gigapath-embeddings' THEN 'prov-gigapath'
        WHEN 'virchow2-embeddings' THEN 'virchow2'
        ELSE builder_id
    END,
    builder_id = CASE builder_id
        WHEN 'prostate-heatmap' THEN 'heatmap'
        WHEN 'episeg-segmentation' THEN 'heatmap'
        WHEN 'gigapath-embeddings' THEN 'tile-embeddings'
        WHEN 'virchow2-embeddings' THEN 'tile-embeddings'
        ELSE builder_id
    END;

-- Output paths moved from builder configuration to the invocation.
UPDATE executions
SET invocation = invocation #- '{resolved_configuration,output_path}'
WHERE jsonb_typeof(invocation -> 'resolved_configuration') = 'object';

ALTER TABLE executions ALTER COLUMN model_id SET NOT NULL;

DROP INDEX executions_active_processing_idx;
CREATE INDEX executions_active_processing_idx ON executions (model_id, lease_expires_at)
    WHERE work_kind = 'processing' AND fencing_token IS NOT NULL;
