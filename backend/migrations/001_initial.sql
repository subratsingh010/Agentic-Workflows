CREATE TABLE IF NOT EXISTS conversation_messages (
  id VARCHAR(36) PRIMARY KEY,
  thread_id VARCHAR(128) NOT NULL,
  role VARCHAR(32) NOT NULL,
  content TEXT NOT NULL,
  extra JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMP NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_conversation_messages_thread_id
  ON conversation_messages(thread_id);

CREATE TABLE IF NOT EXISTS agent_checkpoints (
  thread_id VARCHAR(128) PRIMARY KEY,
  state JSONB NOT NULL,
  updated_at TIMESTAMP NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS audit_events (
  id VARCHAR(36) PRIMARY KEY,
  actor_subject VARCHAR(128),
  action VARCHAR(128) NOT NULL,
  outcome VARCHAR(64) NOT NULL,
  extra JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMP NOT NULL DEFAULT now()
);


CREATE TABLE IF NOT EXISTS pending_actions (
  token VARCHAR(128) PRIMARY KEY,
  thread_id VARCHAR(128) NOT NULL,
  actor_subject VARCHAR(128) NOT NULL,
  intent VARCHAR(64) NOT NULL,
  tool_payload JSONB NOT NULL,
  message TEXT NOT NULL,
  expires_at TIMESTAMP NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_pending_actions_thread_id
  ON pending_actions(thread_id);

CREATE INDEX IF NOT EXISTS ix_pending_actions_actor_subject
  ON pending_actions(actor_subject);

CREATE INDEX IF NOT EXISTS ix_pending_actions_expires_at
  ON pending_actions(expires_at);
