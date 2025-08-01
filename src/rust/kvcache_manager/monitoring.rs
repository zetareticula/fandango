// SPDX-License-Identifier: Apache-2.0
// This file is part of the Zeta Reticula - Fandango  project, which is licensed under the Apache License 2.0.

// This file contains functions to calculate entropy and measure locality of attention data.
// The entropy function calculates the uncertainty in the data distribution,
// while the locality function measures how close values are to the mean of the data.

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