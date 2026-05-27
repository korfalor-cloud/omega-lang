/// Message queue for inter-process communication.

use std::collections::VecDeque;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
pub struct Message {
    pub id: u64,
    pub topic: String,
    pub payload: Vec<u8>,
    pub headers: std::collections::HashMap<String, String>,
    pub timestamp: u64,
    pub delivery_count: u32,
    pub max_retries: u32,
}

impl Message {
    pub fn new(topic: &str, payload: &[u8]) -> Self {
        static mut COUNTER: u64 = 0;
        let id = unsafe {
            COUNTER += 1;
            COUNTER
        };

        Self {
            id,
            topic: topic.to_string(),
            payload: payload.to_vec(),
            headers: std::collections::HashMap::new(),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            delivery_count: 0,
            max_retries: 3,
        }
    }

    pub fn with_header(mut self, key: &str, value: &str) -> Self {
        self.headers.insert(key.to_string(), value.to_string());
        self
    }

    pub fn with_max_retries(mut self, max: u32) -> Self {
        self.max_retries = max;
        self
    }

    pub fn payload_as_string(&self) -> String {
        String::from_utf8_lossy(&self.payload).to_string()
    }

    pub fn is_retryable(&self) -> bool {
        self.delivery_count < self.max_retries
    }
}

#[derive(Debug)]
pub struct MessageQueue {
    queues: std::collections::HashMap<String, VecDeque<Message>>,
    subscribers: std::collections::HashMap<String, Vec<Box<dyn Fn(&Message)>>>,
    dead_letter: Vec<Message>,
    max_queue_size: usize,
}

impl MessageQueue {
    pub fn new() -> Self {
        Self {
            queues: std::collections::HashMap::new(),
            subscribers: std::collections::HashMap::new(),
            dead_letter: Vec::new(),
            max_queue_size: 10000,
        }
    }

    pub fn with_max_queue_size(mut self, size: usize) -> Self {
        self.max_queue_size = size;
        self
    }

    pub fn publish(&mut self, message: Message) -> Result<(), String> {
        let queue = self.queues
            .entry(message.topic.clone())
            .or_insert_with(VecDeque::new);

        if queue.len() >= self.max_queue_size {
            return Err("Queue full".to_string());
        }

        queue.push_back(message.clone());

        // Notify subscribers
        if let Some(handlers) = self.subscribers.get(&message.topic) {
            for handler in handlers {
                handler(&message);
            }
        }

        Ok(())
    }

    pub fn subscribe<F: Fn(&Message) + 'static>(&mut self, topic: &str, handler: F) {
        self.subscribers
            .entry(topic.to_string())
            .or_insert_with(Vec::new)
            .push(Box::new(handler));
    }

    pub fn consume(&mut self, topic: &str) -> Option<Message> {
        self.queues.get_mut(topic)?.pop_front()
    }

    pub fn peek(&self, topic: &str) -> Option<&Message> {
        self.queues.get(topic)?.front()
    }

    pub fn queue_size(&self, topic: &str) -> usize {
        self.queues.get(topic).map_or(0, |q| q.len())
    }

    pub fn total_messages(&self) -> usize {
        self.queues.values().map(|q| q.len()).sum()
    }

    pub fn topics(&self) -> Vec<&str> {
        self.queues.keys().map(|s| s.as_str()).collect()
    }

    pub fn ack(&mut self, topic: &str, message_id: u64) {
        // In a real implementation, would remove from in-flight
    }

    pub fn nack(&mut self, topic: &str, message_id: u64) {
        if let Some(queue) = self.queues.get_mut(topic) {
            if let Some(pos) = queue.iter().position(|m| m.id == message_id) {
                let mut msg = queue.remove(pos).unwrap();
                msg.delivery_count += 1;
                if msg.is_retryable() {
                    queue.push_back(msg);
                } else {
                    self.dead_letter.push(msg);
                }
            }
        }
    }

    pub fn dead_letter_count(&self) -> usize {
        self.dead_letter.len()
    }

    pub fn dead_letters(&self) -> &[Message] {
        &self.dead_letter
    }

    pub fn purge(&mut self, topic: &str) {
        if let Some(queue) = self.queues.get_mut(topic) {
            queue.clear();
        }
    }

    pub fn purge_all(&mut self) {
        self.queues.clear();
    }
}

/// Topic exchange for pub/sub
#[derive(Debug)]
pub struct TopicExchange {
    bindings: std::collections::HashMap<String, Vec<String>>,
}

impl TopicExchange {
    pub fn new() -> Self {
        Self {
            bindings: std::collections::HashMap::new(),
        }
    }

    pub fn bind(&mut self, pattern: &str, queue: &str) {
        self.bindings
            .entry(pattern.to_string())
            .or_insert_with(Vec::new)
            .push(queue.to_string());
    }

    pub fn route(&self, topic: &str) -> Vec<&str> {
        let mut result = Vec::new();
        for (pattern, queues) in &self.bindings {
            if self.matches(pattern, topic) {
                for queue in queues {
                    result.push(queue.as_str());
                }
            }
        }
        result
    }

    fn matches(&self, pattern: &str, topic: &str) -> bool {
        if pattern == "*" || pattern == topic {
            return true;
        }

        if pattern.ends_with(".*") {
            let prefix = &pattern[..pattern.len() - 2];
            return topic.starts_with(prefix);
        }

        if pattern.contains('#') {
            let parts: Vec<&str> = pattern.split('#').collect();
            if parts.len() == 2 {
                return topic.starts_with(parts[0]) && topic.ends_with(parts[1]);
            }
        }

        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_queue() {
        let mut mq = MessageQueue::new();
        let msg = Message::new("test", b"hello");
        mq.publish(msg).unwrap();

        assert_eq!(mq.queue_size("test"), 1);
        let consumed = mq.consume("test").unwrap();
        assert_eq!(consumed.payload_as_string(), "hello");
    }

    #[test]
    fn test_message_with_headers() {
        let msg = Message::new("test", b"data")
            .with_header("content-type", "application/json");
        assert_eq!(msg.headers.get("content-type").unwrap(), "application/json");
    }

    #[test]
    fn test_queue_full() {
        let mut mq = MessageQueue::new().with_max_queue_size(2);
        mq.publish(Message::new("q", b"1")).unwrap();
        mq.publish(Message::new("q", b"2")).unwrap();
        assert!(mq.publish(Message::new("q", b"3")).is_err());
    }

    #[test]
    fn test_nack_retry() {
        let mut mq = MessageQueue::new();
        let msg = Message::new("q", b"test").with_max_retries(2);
        mq.publish(msg).unwrap();

        let consumed = mq.consume("q").unwrap();
        mq.nack("q", consumed.id);
        assert_eq!(mq.queue_size("q"), 1);
    }

    #[test]
    fn test_topic_exchange() {
        let mut exchange = TopicExchange::new();
        exchange.bind("orders.*", "order_queue");
        exchange.bind("payments.*", "payment_queue");

        let routes = exchange.route("orders.created");
        assert!(routes.contains(&"order_queue"));
    }
}
