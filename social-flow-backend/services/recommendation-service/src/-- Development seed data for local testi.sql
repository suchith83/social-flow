-- Development seed data for local testing. IDs use predictable prefixes to make tests deterministic.

BEGIN TRANSACTION;

-- Users
INSERT OR IGNORE INTO users (id,email,display_name,salt,pwd_hash,role,created_at) VALUES
('user_demo_1','alice@example.com','alice','salt','hash','creator',strftime('%s','now')),
('user_demo_2','bob@example.com','bob','salt','hash','user',strftime('%s','now'));

-- Videos
INSERT OR IGNORE INTO videos (id,title,description,uploader_id,duration,s3_key,uploaded_at,status) VALUES
('vid_demo_1','Intro to Social Flow','Welcome video','user_demo_1',120.0,'uploads/vid_demo_1.mp4',strftime('%s','now'),'ready'),
('vid_demo_2','Device Review','Review content','user_demo_2',300.0,'uploads/vid_demo_2.mp4',strftime('%s','now'),'ready');

-- Posts
INSERT OR IGNORE INTO posts (id,user_id,text,media_ref,created_at) VALUES
('post_demo_1','user_demo_1','Hello world from Alice',NULL,strftime('%s','now')),
('post_demo_2','user_demo_2','Bob shares a thought',NULL,strftime('%s','now'));

-- Recommendation feedback
INSERT OR IGNORE INTO recommendation_feedback (id,user_id,item_id,action,timestamp,payload_json,created_at) VALUES
('fb_demo_1','user_demo_2','vid_demo_1','view',strftime('%s','now'), '{"note":"seed view"}',strftime('%s','now'));

-- Analytics events
INSERT OR IGNORE INTO analytics_events (id,name,value,tags_json,timestamp,created_at) VALUES
('evt_demo_1','video.play',1.0,'{\"video_id\":\"vid_demo_1\",\"user_id\":\"user_demo_2\"}',strftime('%s','now'),strftime('%s','now'));

COMMIT;
