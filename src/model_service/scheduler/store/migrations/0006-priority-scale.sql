-- Priorities are LOW 0, NORMAL 1, HIGH 2; 0005 stored NORMAL as 0 and the top
-- level as 100.
-- depends: 0005-priority

UPDATE executions SET priority = CASE WHEN priority >= 100 THEN 2 ELSE 1 END;
ALTER TABLE executions ALTER COLUMN priority SET DEFAULT 1;
