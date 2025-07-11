use crate::ir_generator::profiling::Profiler;
use crate::kvcache_manager::entropy_heuristic;
use crate::kvcache_manager::precision_plan::PrecisionPlanResult;
use crate::kvcache_manager::profile::Profile;

#[derive(Debug, Clone)]
pub struct PrecisionPlan {
    pub name: String,
    pub description: String,
}

#[derive(Debug, Clone)]
pub struct PrecisionPlanResult {
    pub precision: String,
    pub sparsity: f32,
}

impl PrecisionPlan {
    pub fn new(name: &str, description: &str) -> Self {
        PrecisionPlan {
            name: name.to_string(),
            description: description.to_string(),
        }
    }
}



impl PrecisionPlan {
    pub fn compute_plan(&self, layer_data: &[f32], profiler: &mut Profiler) -> PrecisionPlanResult {
        let entropy = entropy_heuristic::calculate_entropy(layer_data);
        let sensitivity = self.estimate_sensitivity(layer_data);
        let profile = self.profile_layer(layer_data, profiler);

        let precision = match (entropy, sensitivity, profile.load) {
            (e, _, _) if e > 0.8 => "float16".to_string(),
            (_, _, load) if load < 0.3 => "int4".to_string(),
            _ => {
                if layer_data.is_empty() { "fp8" } else { "int8" }.to_string()
            }
        };

        PrecisionPlanResult {
            precision,
            sparsity: if entropy < 0.2 { 0.9 } else { 0.3 },
        }
    }

    fn estimate_sensitivity(&self, _data: &[f32]) -> f32 {
        0.4 // Placeholder
    }

    fn profile_layer(&self, data: &[f32], profiler: &mut Profiler) -> Profile {
        let load = profiler.profile_layer("sample_layer", || {
            // Simulate computation
            for _ in 0..1000 {
                let _ = data.iter().sum::<f32>();
            }
        });
        Profile { load: load / 10.0 } // Normalize load
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir_generator::profiling::Profiler;

    #[test]
    fn test_precision_plan_computation() {
        let plan = PrecisionPlan::new("Test Plan", "A test precision plan");
        let layer_data = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let mut profiler = Profiler::new();

        let result = plan.compute_plan(&layer_data, &mut profiler);
        assert_eq!(result.precision, "int8");
        assert!(result.sparsity >= 0.3);
    }
}