use tokio::sync::mpsc;
use crate::kvcache_manager::KVCacheManager;

pub struct CompactionAgent {
    rx: mpsc::Receiver<CompactionRequest>,
    kvcache_mgr: KVCacheManager,
}

impl CompactionAgent {
    pub fn new(rx: mpsc::Receiver<CompactionRequest>, kvcache_mgr: KVCacheManager) -> Self {
        CompactionAgent { rx, kvcache_mgr }
    }

    pub async fn run(&mut self) {
        while let Some(request) = self.rx.recv().await {
            self.kvcache_mgr.perform_remote_compaction(&request.job_id, &request.data_access).await;
        }
    }
}

async fn perform_remote_compaction(&self, job_id: &str, data_access: &[String]) {
    // Simulate remote compaction (e.g., fetch from disaggregated storage)
    println!("Compacting job {} remotely", job_id);
}