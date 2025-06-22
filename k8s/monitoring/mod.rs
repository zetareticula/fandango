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
        "fandango_processing_latency_seconds",
        "Latency of processing operations",
        vec![0.001, 0.01, 0.1, 1.0, 10.0]
    ).unwrap()
});

pub static SPECULATIVE_ACCEPTED: Lazy<IntCounter> = Lazy::new(|| {
    register_int_counter!("fandango_speculative_accepted_total", "Total number of accepted speculative tokens").unwrap()
});

pub static SPECULATIVE_REJECTED: Lazy<IntCounter> = Lazy::new(|| {
    register_int_counter!("fandango_speculative_rejected_total", "Total number of rejected speculative tokens").unwrap()
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

pub fn record_speculative_accepted() {
    SPECULATIVE_ACCEPTED.inc();
}

pub fn record_speculative_rejected() {
    SPECULATIVE_REJECTED.inc();
}