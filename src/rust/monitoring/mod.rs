use prometheus::{register_int_counter, register_histogram, IntCounter, Histogram};
use once_cell::sync::Lazy;

pub static CACHE_HITS: Lazy<IntCounter> = Lazy::new(|| {
    register_int_counter!("fandango_cache_hits_total", "Total number of cache hits").unwrap()
});

pub static CACHE_MISSES: Lazy<IntCounter> = Lazy::new(|| {
    register_int_counter!("fandango_cache_misses_total", "Total number of cache misses").unwrap()
});

pub static EVICTION_EVENTS: Lazy<IntCounter> = Lazy::new(|| {
    register_int_counter!("fandango_cache_eviction_total", "Total number of cache evictions").unwrap()
});

pub static LATENCY: Lazy<Histogram> = Lazy::new(|| {
    register_histogram!(
        "fandango_processing_latency_ms",
        "Latency of processing operations in ms",
        vec![1.0, 10.0, 20.0, 30.0, 40.0]
    ).unwrap()
});

pub static KV_CACHE_SIZE: Lazy<Histogram> = Lazy::new(|| {
    register_histogram!(
        "fandango_kv_cache_size_bytes",
        "KV cache size in bytes",
        vec![1_000_000, 10_000_000, 100_000_000, 1_000_000_000]
    ).unwrap()
});

pub fn record_cache_hit() {
    CACHE_HITS.inc();
}

pub fn record_cache_miss() {
    CACHE_MISSES.inc();
}

pub fn record_eviction() {
    EVICTION_EVENTS.inc();
}

pub fn record_latency(latency: f64) {
    LATENCY.observe(latency);
}

pub fn record_kv_cache_size(size: f64) {
    KV_CACHE_SIZE.observe(size);
}