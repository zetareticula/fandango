// SPDX-License-Identifier: Apache-2.0
// This file is part of the Zeta Reticula - Fandango  project, which is licensed under the Apache License 2.0.

// This file contains functions to calculate entropy and measure locality of attention data.
// The entropy function calculates the uncertainty in the data distribution,
// while the locality function measures how close values are to the mean of the data.

use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
use std::collections::VecDeque;

impl PrefetchManager {
    pub async fn prefetch_kv_caches_with_neurons(&self, active_neurons: Vec<usize>) {
        let buffer = self.buffer.lock().unwrap();
        for &neuron_idx in &active_neurons {
            let job_id = format!("job_{}", neuron_idx);
            if buffer.get_kv_cache(&job_id).is_none() {
                let kv_data = self.fetch_from_disk(&job_id).await;
                if let Ok(data) = kv_data {
                    let mut buffer = self.buffer.lock().unwrap();
                    buffer.add_kv_cache(job_id, data);
                }
            }
        }
    }
}