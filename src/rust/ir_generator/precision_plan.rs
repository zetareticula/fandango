use crate::kvcache_manager::entropy_heuristic;

impl PrecisionPlan {
    pub fn compute_plan(&self, layer_data: &[f32]) -> PrecisionPlanResult {
        let entropy = entropy_heuristic::calculate_entropy(layer_data);
        let sensitivity = self.estimate_sensitivity(layer_data);
        let profile = self.profile_layer(layer_data);

        let precision = match (entropy, sensitivity, profile.load) {
            (e, _, _) if e > 0.8 => "float16".to_string(),
            (_, _, load) if load < 0.3 => "int4".to_string(),
            _ => {
                if layer_data.is_empty() { "int8" } else { "bf16" }.to_string()
            }
        };

        PrecisionPlanResult {
            precision,
            sparsity: if entropy < 0.2 { 0.9 } else { 0.3 },
        }
    }
}