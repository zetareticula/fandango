use crate::fused_attention_kernels::sparsity_manager::SparsityManager;
use crate::fused_attention_kernels::memory_layout::FFNMemoryLayout;

impl FusedAttention {
    pub fn process_mermaid_flow(&mut self, input: Vec<f32>) -> Result<Vec<f32>, AttentionError> {
        let tensor = Tensor::from_vec(input, (input.len(), 1), &self.device)?
            .to_dtype(DType::F32).map_err(AttentionError::CandleError)?;
        let mut output = tensor.clone();

        // Initialize sparsity manager and memory layout
        let mut sparsity_mgr = SparsityManager::new(self.device.clone(), 3)?; // Window size 3
        let mut memory_layout = FFNMemoryLayout::new(1024)?;

        output = self.embedding_layer(&output)?;
        output = self.attention_qkv(&output)?;
        self.kvcache_mgr.update_precision(&output.to_vec1::<f32>()?.map_err(AttentionError::CandleError)?, 0.5)
            .await.map_err(|_| AttentionError::KVCacheUpdate)?;
        output = self.sparse_attention(&output)?;
        output = sparsity_mgr.process_ffn(&output)?; // Use sparsity manager for FFN
        output = self.output_layer_fp8(&output)?;

        Ok(output.to_vec1::<f32>().map_err(AttentionError::CandleError)?)
    }
}