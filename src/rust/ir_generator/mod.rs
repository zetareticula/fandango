use crate::ir_generator::precision_plan::PrecisionPlan;
use crate::fused_attention_kernels::FusedAttention;

pub struct IRGenerator {
    precision_plan: PrecisionPlan,
    execution_graph: Vec<LayerNode>,
}

#[derive(Clone)]
struct LayerNode {
    name: String,
    precision: String,
    next: Option<String>,
}

impl IRGenerator {
    pub fn new() -> Self {
        IRGenerator {
            precision_plan: PrecisionPlan::new(),
            execution_graph: Vec::new(),
        }
    }

    pub fn build_ir(&mut self, input_data: &[f32]) {
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
            let precision = self.precision_plan.compute_plan(data).precision;
            self.execution_graph.push(LayerNode {
                name: name.to_string(),
                precision,
                next,
            });
        }

        self.schedule_execution();
    }

    fn schedule_execution(&self) {
        for node in &self.execution_graph {
            println!("Scheduling: {} (Precision: {}) -> {:?}", node.name, node.precision, node.next);
            if node.precision != "float32" {
                println!("Dequant -> Compute -> Requant for {}", node.name);
            }
        }
    }
}