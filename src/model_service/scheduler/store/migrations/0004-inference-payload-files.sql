-- Raw-inference request and response bytes move from the database to files on
-- the inference storage volume; executions keep their relative paths and sizes.
-- The stored bytes cannot be carried over to files, so existing inference
-- executions are removed. They are short-lived tiles (24 h retention by
-- default); drain scheduled raw inference before upgrading to keep results.
-- depends: 0003-raw-inference

DELETE FROM executions WHERE operation = 'inference';
DROP TABLE inference_payloads;

ALTER TABLE executions
    ADD COLUMN request_path TEXT,
    ADD COLUMN request_size BIGINT,
    ADD COLUMN response_path TEXT,
    ADD COLUMN response_size BIGINT,
    ADD CHECK ((request_path IS NULL) = (request_size IS NULL)),
    ADD CHECK ((response_path IS NULL) = (response_size IS NULL)),
    ADD CHECK (operation = 'inference' OR (request_path IS NULL AND response_path IS NULL));
