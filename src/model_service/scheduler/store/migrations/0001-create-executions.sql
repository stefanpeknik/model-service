-- Create the durable workflow execution queue.
-- depends:

CREATE TABLE executions (
    id UUID PRIMARY KEY,
    provider TEXT NOT NULL,
    submission_key TEXT NOT NULL,
    request_fingerprint TEXT NOT NULL,
    workflow_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')
    ),
    invocation JSONB NOT NULL,
    protected_context BYTEA NOT NULL,
    deadline_at TIMESTAMPTZ NOT NULL,
    terminal_report_deadline_at TIMESTAMPTZ,
    work_kind TEXT CHECK (
        work_kind IS NULL OR work_kind IN (
            'processing', 'provider_finish',
            'provider_failure_report', 'provider_cancellation_report'
        )
    ),
    next_attempt_at TIMESTAMPTZ,
    processing_attempts INTEGER NOT NULL DEFAULT 0 CHECK (processing_attempts >= 0),
    processing_rejections INTEGER NOT NULL DEFAULT 0 CHECK (processing_rejections >= 0),
    finish_attempts INTEGER NOT NULL DEFAULT 0 CHECK (finish_attempts >= 0),
    finish_mutation_started BOOLEAN NOT NULL DEFAULT FALSE,
    lease_owner TEXT,
    fencing_token UUID,
    lease_expires_at TIMESTAMPTZ,
    result JSONB,
    error_code TEXT,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    UNIQUE (provider, submission_key),
    CHECK ((error_code IS NULL) = (error_message IS NULL)),
    CHECK (
        (fencing_token IS NULL AND lease_owner IS NULL AND lease_expires_at IS NULL)
        OR
        (fencing_token IS NOT NULL AND lease_owner IS NOT NULL AND lease_expires_at IS NOT NULL)
    ),
    CHECK (
        result IS NULL
        OR work_kind = 'provider_finish'
        OR status IN ('succeeded', 'failed', 'cancelled')
    )
);

CREATE INDEX executions_due_idx ON executions (next_attempt_at, created_at)
    WHERE work_kind IS NOT NULL;
CREATE INDEX executions_active_processing_idx ON executions (workflow_id, lease_expires_at)
    WHERE work_kind = 'processing' AND fencing_token IS NOT NULL;
