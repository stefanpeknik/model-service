-- An invocation's `resolved_configuration` becomes `configuration`: the settings
-- that differ from the builder's defaults, applied by each attempt.
-- depends: 0007-drop-raw-inference

UPDATE executions
SET invocation = (invocation - 'resolved_configuration')
    || jsonb_build_object('configuration', invocation -> 'resolved_configuration')
WHERE invocation -> 'resolved_configuration' IS NOT NULL;
