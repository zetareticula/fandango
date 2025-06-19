use crate::kvcache_manager::monitoring::Monitoring;

impl KVCacheManager {
    pub async fn update_precision(&mut self, attention_data: &[f32]) {
        let monitoring = Monitoring::new().await.unwrap();
        let system_load = monitoring.get_system_load().await;

        let entropy = calculate_entropy(attention_data);
        let locality = measure_locality(attention_data);

        self.load_level = system_load;

        self.current_precision = match (entropy, locality, system_load) {
            (_, _, load) if load < 0.3 => Precision::Int4,
            (_, _, load) if load >= 0.7 => Precision::Int8,
            (e, l, _) if e > 0.8 || l < 0.2 => Precision::Float16,
            _ => Precision::Int8,
        };

        self.deduplicate_and_quantize(attention_data);
        log_precision_change(system_load, self.current_precision);
    }
}

fn log_precision_change(load: f32, precision: Precision) {
    println!("Load: {:.2}, Precision switched to: {:?}", load, precision);
}