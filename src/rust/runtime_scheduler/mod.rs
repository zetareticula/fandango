use crate::kvcache_manager::KVCacheManager;
use crate::cognitive_modeling::{CognitiveModel, MCMCSearch};
use tokio::sync::mpsc;

pub struct RuntimeScheduler {
    kv_cache_mgr: KVCacheManager,
    job_queue: Vec<String>,
    tx: mpsc::Sender<String>,
    cognitive_model: CognitiveModel,
    mcmc_search: MCMCSearch,
}

impl RuntimeScheduler {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::channel(32);
        let mut kv_cache_mgr = KVCacheManager::new();
        kv_cache_mgr.prefetch_mgr.rx = rx;
        let cognitive_model = CognitiveModel::new();
        let mcmc_search = MCMCSearch::new();
        RuntimeScheduler {
            kv_cache_mgr,
            job_queue: Vec::new(),
            tx,
            cognitive_model,
            mcmc_search,
        }
    }

    pub async fn schedule(&mut self, job_id: &str) {
        self.job_queue.push(job_id.to_string());
        self.cognitive_model.add_evidence(job_id.to_string(), 0.5); // Example likelihood
        self.mcmc_search.search(100); // Run MCMC to update theories
        self.cognitive_model.update_theory_space();

        if let Err(e) = self.tx.send(job_id.to_string()).await {
            eprintln!("Failed to send job to prefetch manager: {}", e);
        }
        self.kv_cache_mgr.update_precision(&vec![0.0; 64], 0.5).await;
    }
}


