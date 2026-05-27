/// Session management for web applications.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
pub struct Session {
    id: String,
    data: HashMap<String, String>,
    created_at: u64,
    expires_at: u64,
    max_age: u64,
}

impl Session {
    pub fn new() -> Self {
        let now = current_timestamp();
        Self {
            id: generate_session_id(),
            data: HashMap::new(),
            created_at: now,
            expires_at: now + 3600, // 1 hour default
            max_age: 3600,
        }
    }

    pub fn with_max_age(mut self, seconds: u64) -> Self {
        let now = current_timestamp();
        self.max_age = seconds;
        self.expires_at = now + seconds;
        self
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.data.get(key).map(|s| s.as_str())
    }

    pub fn set(&mut self, key: &str, value: &str) {
        self.data.insert(key.to_string(), value.to_string());
    }

    pub fn remove(&mut self, key: &str) -> Option<String> {
        self.data.remove(key)
    }

    pub fn has(&self, key: &str) -> bool {
        self.data.contains_key(key)
    }

    pub fn clear(&mut self) {
        self.data.clear();
    }

    pub fn is_expired(&self) -> bool {
        current_timestamp() >= self.expires_at
    }

    pub fn refresh(&mut self) {
        let now = current_timestamp();
        self.expires_at = now + self.max_age;
    }

    pub fn created_at(&self) -> u64 {
        self.created_at
    }

    pub fn expires_at(&self) -> u64 {
        self.expires_at
    }

    pub fn keys(&self) -> Vec<&str> {
        self.data.keys().map(|s| s.as_str()).collect()
    }

    pub fn values(&self) -> Vec<&str> {
        self.data.values().map(|s| s.as_str()).collect()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn to_cookie(&self) -> String {
        format!("session_id={}; Max-Age={}; Path=/; HttpOnly; SameSite=Strict",
            self.id, self.max_age)
    }
}

/// Session store
#[derive(Debug)]
pub struct SessionStore {
    sessions: HashMap<String, Session>,
    default_max_age: u64,
}

impl SessionStore {
    pub fn new() -> Self {
        Self {
            sessions: HashMap::new(),
            default_max_age: 3600,
        }
    }

    pub fn with_default_max_age(mut self, seconds: u64) -> Self {
        self.default_max_age = seconds;
        self
    }

    pub fn create(&mut self) -> &Session {
        let session = Session::new().with_max_age(self.default_max_age);
        let id = session.id().to_string();
        self.sessions.insert(id.clone(), session);
        self.sessions.get(&id).unwrap()
    }

    pub fn get(&self, id: &str) -> Option<&Session> {
        self.sessions.get(id).filter(|s| !s.is_expired())
    }

    pub fn get_mut(&mut self, id: &str) -> Option<&mut Session> {
        if self.sessions.get(id).map_or(false, |s| !s.is_expired()) {
            self.sessions.get_mut(id)
        } else {
            None
        }
    }

    pub fn destroy(&mut self, id: &str) -> bool {
        self.sessions.remove(id).is_some()
    }

    pub fn cleanup(&mut self) {
        self.sessions.retain(|_, session| !session.is_expired());
    }

    pub fn count(&self) -> usize {
        self.sessions.len()
    }

    pub fn from_cookie(cookie: &str) -> Option<&str> {
        for part in cookie.split(';') {
            let part = part.trim();
            if let Some(value) = part.strip_prefix("session_id=") {
                return Some(value.trim());
            }
        }
        None
    }
}

fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn generate_session_id() -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    current_timestamp().hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

/// Flash messages for session-based notifications
pub struct FlashMessages {
    messages: Vec<FlashMessage>,
}

#[derive(Debug, Clone)]
pub struct FlashMessage {
    pub level: FlashLevel,
    pub text: String,
}

#[derive(Debug, Clone)]
pub enum FlashLevel {
    Success,
    Info,
    Warning,
    Error,
}

impl FlashMessages {
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
        }
    }

    pub fn success(&mut self, text: &str) {
        self.messages.push(FlashMessage {
            level: FlashLevel::Success,
            text: text.to_string(),
        });
    }

    pub fn info(&mut self, text: &str) {
        self.messages.push(FlashMessage {
            level: FlashLevel::Info,
            text: text.to_string(),
        });
    }

    pub fn warning(&mut self, text: &str) {
        self.messages.push(FlashMessage {
            level: FlashLevel::Warning,
            text: text.to_string(),
        });
    }

    pub fn error(&mut self, text: &str) {
        self.messages.push(FlashMessage {
            level: FlashLevel::Error,
            text: text.to_string(),
        });
    }

    pub fn messages(&self) -> &[FlashMessage] {
        &self.messages
    }

    pub fn has_messages(&self) -> bool {
        !self.messages.is_empty()
    }

    pub fn drain(&mut self) -> Vec<FlashMessage> {
        std::mem::take(&mut self.messages)
    }

    pub fn to_html(&self) -> String {
        self.messages.iter().map(|msg| {
            let class = match msg.level {
                FlashLevel::Success => "flash-success",
                FlashLevel::Info => "flash-info",
                FlashLevel::Warning => "flash-warning",
                FlashLevel::Error => "flash-error",
            };
            format!("<div class=\"{}\">{}</div>", class, msg.text)
        }).collect::<Vec<_>>().join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_operations() {
        let mut session = Session::new();
        session.set("user_id", "42");
        assert_eq!(session.get("user_id"), Some("42"));
        session.remove("user_id");
        assert!(session.get("user_id").is_none());
    }

    #[test]
    fn test_session_store() {
        let mut store = SessionStore::new();
        store.create();
        assert_eq!(store.count(), 1);
    }

    #[test]
    fn test_session_cookie() {
        let session = Session::new();
        let cookie = session.to_cookie();
        assert!(cookie.contains("session_id="));
        assert!(cookie.contains("HttpOnly"));
    }

    #[test]
    fn test_from_cookie() {
        let cookie = "session_id=abc123; Path=/";
        assert_eq!(SessionStore::from_cookie(cookie), Some("abc123"));
    }

    #[test]
    fn test_flash_messages() {
        let mut flash = FlashMessages::new();
        flash.success("Created!");
        flash.error("Failed!");
        assert!(flash.has_messages());
        assert_eq!(flash.messages().len(), 2);
    }

    #[test]
    fn test_flash_html() {
        let mut flash = FlashMessages::new();
        flash.info("Hello");
        let html = flash.to_html();
        assert!(html.contains("flash-info"));
    }
}
