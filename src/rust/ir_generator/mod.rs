use crate::ir_generator::profiling::Profiler;
use crate::ir_generator::precision_plan::PrecisionPlan;


impl IRGenerator {
    pub fn build_ir(&mut self, input_data: &[f32]) {
        let mut profiler = Profiler::new();
        let layers = vec![
            ("Token", "Input Tokens", vec![]),
            ("E", "Embedding Layer FP8", input_data.to_vec()),
            ("A1", "Attention (QKV) - INT4", vec![]),
            ("UVC", "LWCached Quantized Storage (INT4/INT8)", vec![]),
            ("A2", "Sparse Attention - INT4/INT8", vec![]),
            ("FF", "FeedForward - INT8 or BF16", vec![]),
            ("O", "Output Layer - FP8", vec![]),
        ];

        for (i, (name, desc, data)) in layers.iter().enumerate() {
            let next = layers.get(i + 1).map(|(next_name, _, _)| next_name.to_string());
            let precision = self.precision_plan.compute_plan(data, &mut profiler).precision;
            profiler.profile_layer(name, || {
                // Simulate layer execution
            });
            self.execution_graph.push(LayerNode {
                name: name.to_string(),
                precision,
                next,
            });
        }

        self.schedule_execution();
        println!("Profiling Results: {:?}", profiler.get_profile());
    }
}