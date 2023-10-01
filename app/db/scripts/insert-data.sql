-- USERS
INSERT INTO users (uuid, username, email, password) VALUES
    (uuid_generate_v4(), 'Pieter', 'pieter@gmail.com', 'pieterpass123'),
    (uuid_generate_v4(), 'Pompon', 'pompon@gmail.com', 'pomponpass123'),
    (uuid_generate_v4(), 'Pralek', 'pralek@gmail.com', 'pralekpass123');



-- SUBMITS
INSERT INTO submits (uuid, coordinates, reported_animal, report_ts) VALUES
    ('f5ae175b-ddc0-4d3a-a33d-fdc8ce054308', ARRAY [50.049688, 19.944548], 'fox', '2023-09-30T16:51:25+00:00'),
    ('791d9279-3c6b-4136-80d9-790d14e74e35', ARRAY [50.012108, 20.985848], 'hog', '2023-09-30T12:51:25+00:00'),
    ('e1aab05d-02ed-42b4-a6f7-19d220bfef1e', ARRAY [50.041188, 21.999128], 'dog', '2023-09-30T10:41:25+00:00');



-- CAMERAS
INSERT INTO cameras (uuid, coordinates, address, active, url) VALUES
    ('bca6752d-62ad-4dd5-b789-139869e3801f', ARRAY [50.049683, 19.944544], 'Somewhere-krakow 12', true, ''),
    ('7f05cfff-532d-4dd3-af3b-cd0696ea4f28', ARRAY [50.012100, 20.985842], 'Somewhere-tarnow 34', true, ''),
    ('4d95487b-9c85-440b-9394-fce8a7083785', ARRAY [50.041187, 21.999121], 'Somewhere-rzeszow 56', true, ''),
    ('4d95487b-9c85-440b-9394-cd0696ea4f28', ARRAY [48.021187, 24.999121], 'Nowy Targ 53', true, 'https://hstream1.webcamera.pl/nowytarg_cam_0f85a4/nowytarg_cam_0f85a4.stream/'),
    ('4457dca0-ce31-47c7-a1ed-c0e17fce7891', ARRAY [53.428543, 14.552812], 'Somewhere-szczecin 98', false, '');



-- DETECTIONS
INSERT INTO detections (uuid, detected_animal, confidence, detection_ts, submit_id) VALUES
    (uuid_generate_v4(), 'fox', 0.9, '2023-09-29T16:31:25+00:00', 'f5ae175b-ddc0-4d3a-a33d-fdc8ce054308');

INSERT INTO detections (uuid, detected_animal, confidence, detection_ts, camera_id) VALUES
    (uuid_generate_v4(), 'dog', 0.2, '2023-09-30T10:41:25+00:00', 'bca6752d-62ad-4dd5-b789-139869e3801f'),
    (uuid_generate_v4(), 'hog', 0.75, '2023-09-30T12:51:25+00:00', '7f05cfff-532d-4dd3-af3b-cd0696ea4f28');