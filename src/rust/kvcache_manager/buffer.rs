use std::collections::VecDeque;

pub struct MemoryBuffer {
    capacity: usize,
    used: usize,
    kv_caches: VecDeque<(String, Vec<u8>)>, // (Job ID, KV Cache Data)
}

impl MemoryBuffer {
    pub fn new(capacity: usize) -> Self {
        MemoryBuffer {
            capacity,
            used: 0,
            kv_caches: VecDeque::new(),
        }
    }

    pub fn add_kv_cache(&mut self, job_id: String, data: Vec<u8>) -> bool {
        let size = data.len();
        if self.used + size <= self.capacity {
            self.kv_caches.push_back((job_id, data));
            self.used += size;
            true
        } else {
            false
        }
    }

    pub fn get_kv_cache(&self, job_id: &str) -> Option<Vec<u8>> {
        self.kv_caches.iter().find(|(id, _)| id == job_id).map(|(_, data)| data.clone())
    }
}