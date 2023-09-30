-- USERS
INSERT INTO users (uuid, username, email, password) VALUES
    (uuid_generate_v4(), 'Pieter', 'pieter@gmail.com', 'pieterpass123'),
    (uuid_generate_v4(), 'Pompon', 'pompon@gmail.com', 'pomponpass123'),
    (uuid_generate_v4(), 'Pralek', 'pralek@gmail.com', 'pralekpass123');



-- SUBMITS
INSERT INTO submits (coordinates, reported_animal, report_ts) VALUES
    (ARRAY [50.049688, 19.944548], 'fox', '2023-09-30T16:51:25+00:00'),
    (ARRAY [50.012108, 20.985848], 'hog', '2023-09-30T12:51:25+00:00'),
    (ARRAY [50.041188, 21.999128], 'dog', '2023-09-30T10:41:25+00:00');



-- CAMERAS
INSERT INTO cameras (uuid, coordinates, address, active, url) VALUES
    (uuid_generate_v4(), ARRAY [50.049683, 19.944544], 'Somewhere-krakow 12', true, 'http://krakow-camera.pl'),
    (uuid_generate_v4(), ARRAY [50.012100, 20.985842], 'Somewhere-tarnow 34', true, 'http://tarnow-camera.pl'),
    (uuid_generate_v4(), ARRAY [50.041187, 21.999121], 'Somewhere-rzeszow 56', true, 'http://rzeszow-camera.pl'),
    (uuid_generate_v4(), ARRAY [53.428543, 14.552812], 'Somewhere-szczecin 98', false, 'http://szczecin-camera.pl');



-- DETECTIONS
INSERT INTO detections (uuid, detected_animal, confidence, detection_ts, submit_id) VALUES
    (uuid_generate_v4(), 'fox', 0.9, '2023-09-29T16:31:25+00:00', 1);

INSERT INTO detections (uuid, detected_animal, confidence, detection_ts, camera_id) VALUES
    (uuid_generate_v4(), 'hog', 0.75, '2023-09-30T12:51:25+00:00', <uuid>),
    (uuid_generate_v4(), 'dog', 0.2, '2023-09-30T10:41:25+00:00', <uuid>);