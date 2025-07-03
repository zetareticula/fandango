use candle_core::{Tensor, Device, Result};
use std::collections::VecDeque;
use crate::fused_attention_kernels::sparsity_manager::SparsityManager;
use crate::monitoring;
use crate::models::distilled_opt_config::DistilledOptConfig;

pub struct SpeculativeDecoder {
    sparsity_mgr: SparsityManager,
    draft_lambda: usize,
    window_size_k: usize,
    acceptance_ratio: f32,
    device: Device,
    token_buffer: VecDeque<(usize, Tensor)>,
    draft_model: OptModel,
    memory_mgr: MemoryManager,
}

impl SpeculativeDecoder {
    pub fn new(device: Device, draft_lambda: usize, window_size_k: usize, acceptance_ratio: f32) -> Result<Self> {
        let sparsity_mgr = SparsityManager::new(device.clone(), 3, 10)?;
        let distilled_config = DistilledOptConfig::new(device.clone())?;
        let draft_model = distilled_config.build_model()?;
        let memory_mgr = MemoryManager::new(device.clone())?;
        Ok(SpeculativeDecoder {
            sparsity_mgr,
            draft_lambda: 4, // Fixed to 4 per ablation
            window_size_k,
            acceptance_ratio,
            device,
            token_buffer: VecDeque::new(),
            draft_model,
            memory_mgr,
        })
    }

    pub fn decode(&mut self, input_tokens: &Tensor) -> Result<Tensor> {
        let start = std::time::Instant::now();
        let mut verified_tokens = Vec::new();

        // Optimize DRAM usage for long generations
        let dram_params = self.memory_mgr.adjust_dram_usage(0.75)?; // 75% in DRAM per Figure 7
        let draft_tokens = self.generate_draft_tokens(input_tokens, self.draft_lambda)?;
        self.token_buffer.extend((0..draft_tokens.dim(0)?).map(|i| (i, draft_tokens.narrow(0, i, 1)?)));

        let optimal_window = (self.acceptance_ratio * (self.draft_lambda + 1) as f32).round() as usize;
        let optimal_window = optimal_window.min(self.window_size_k);

        while !self.token_buffer.is_empty() {
            let (token_idx, token) = self.token_buffer.pop_front().unwrap();
            if self.verify_token(&token)? {
                verified_tokens.push(token);
                if verified_tokens.len() >= optimal_window {
                    break;
                }
            } else if token_idx + 1 < self.draft_lambda {
                self.token_buffer.push_back((token_idx + 1, token));
            }
        }

        let verified_tensor = Tensor::cat(&verified_tokens, 0)?;
        let output = self.sparsity_mgr.process_ffn(&verified_tensor)?;

        let latency = start.elapsed().as_secs_f64() * 1000.0; // ms
        monitoring::record_latency(latency); // Expect ~1.4x speedup
        Ok(output)
    }

    fn generate_draft_tokens(&self, input: &Tensor, lambda: usize) -> Result<Tensor> {
        let logits = self.draft_model.forward(input, None)?;
        let next_tokens = candle_core::ops::argmax(&logits, -1)?.narrow(0, 0, lambda)?;
        Ok(next_tokens)
    }

    fn verify_token(&self, token: &Tensor) -> Result<bool> {
        let result = candle_core::ops::mean(token)?.to_scalar::<f32>()? > 0.5;
        if result {
            monitoring::record_speculative_accepted();
        } else {
            monitoring::record_speculative_rejected();
        }
        Ok(result)
    }
}