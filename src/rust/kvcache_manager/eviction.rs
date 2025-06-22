use std::sync::{Arc, Mutex};

pub struct EvictionManager {
    buffer: Arc<Mutex<MemoryBuffer>>,
    eviction_threshold: usize,
}

impl EvictionManager {
    pub fn new(buffer: Arc<Mutex<MemoryBuffer>>) -> Self {
        EvictionManager {
            buffer,
            eviction_threshold: 512 * 1024 * 1024, // 512MB threshold
        }
    }

    pub fn check_eviction_threshold(&self) {
        let buffer = self.buffer.lock().unwrap();
        if buffer.used > self.eviction_threshold {
            self.evict_kv_cache();
        }
    }

    fn evict_kv_cache(&self) {
        let mut buffer = self.buffer.lock().unwrap();
        if let Some((job_id, data)) = buffer.kv_caches.pop_front() {
            buffer.used -= data.len();
            // Simulate writing to disk
            println!("Evicting KV cache for job {} to disk", job_id);
        }
    }
}