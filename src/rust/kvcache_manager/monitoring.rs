use prometheus::{register_int_counter, IntCounter};

lazy_static::lazy_static! {
    static ref KV_CACHE_HITS: IntCounter = register_int_counter!(
        "kvcache_deduplication_hits_total",
        "Total number of KVCache deduplication hits"
    ).unwrap();
}

impl Monitoring {
    pub fn log_deduplication_hit(&self) {
        KV_CACHE_HITS.inc();
    }
}