CREATE TABLE IF NOT EXISTS users (
    uuid UUID PRIMARY KEY,
    username VARCHAR(64) UNIQUE NOT NULL,
    email VARCHAR(128) UNIQUE NOT NULL,
    password VARCHAR(64) NOT NULL
--     role VARCHAR(64) NOT NULL,
--     table_constraints

);

CREATE TABLE IF NOT EXISTS cameras (
    uuid UUID PRIMARY KEY,
    coordinates REAL [] NOT NULL,
    address VARCHAR(128),
    active BOOLEAN NOT NULL,
    url VARCHAR(255) NOT NULL
);

CREATE TABLE IF NOT EXISTS submits (
    uuid UUID PRIMARY KEY,
    coordinates REAL [] NOT NULL,
    reported_animal VARCHAR(64),
    report_ts TIMESTAMP NOT NULL
);

CREATE  TABLE IF NOT EXISTS detections (
    uuid UUID PRIMARY KEY,
    detected_animal VARCHAR(64) NOT NULL,
    confidence REAL NOT NULL,
    detection_ts TIMESTAMP NOT NULL,
    camera_id UUID,
    submit_id UUID,

    FOREIGN KEY (camera_id) REFERENCES cameras(uuid),
    FOREIGN KEY (submit_id) REFERENCES submits(uuid)
);