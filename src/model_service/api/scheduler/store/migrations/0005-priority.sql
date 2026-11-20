-- Due work is claimed by priority first (higher first), then as before.
-- depends: 0004-inference-payload-files

ALTER TABLE executions ADD COLUMN priority INTEGER NOT NULL DEFAULT 0;
