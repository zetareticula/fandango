use std::collections::HashMap;
use tokio::sync::mpsc;
use candle_core::{Device, Result, Tensor, DType};
use crate::{
    nebula_integration::nebula_store::NebulaStore,
    kvcache_manager::KVCacheManager,
    caas_lsm::{CompactionRequest, ControlPlane}
};

pub struct ResourceMonitor {
    available_csas: Vec<String>,
    load_levels: HashMap<String, f32>,
}

impl ResourceMonitor {
    pub fn new() -> Self {
        ResourceMonitor {
            available_csas: vec!["csa1".to_string()],
            load_levels: HashMap::new(),
        }
    }
}

impl ResourceMonitor {
    pub fn update_load(&mut self, csa: &str, load: f32) {
        self.load_levels.insert(csa.to_string(), load);
    }
}



pub struct CompactionAgent {
    rx: mpsc::Receiver<CompactionRequest>,
    kvcache_mgr: KVCacheManager,
    nebula_store: NebulaStore,
    load: f32,
    max_load: f32,
}

pub struct CompactionAgent {
    rx: mpsc::Receiver<CompactionRequest>,
    kvcache_mgr: KVCacheManager,
    nebula_store: NebulaStore,
    load: f32,
    max_load: f32,
    control_plane_tx: mpsc::Sender<CompactionRequest>,
}

impl CompactionAgent {
    pub fn new(
        rx: mpsc::Receiver<CompactionRequest>,
        kvcache_mgr: KVCacheManager,
        device: Device,
        control_plane_tx: mpsc::Sender<CompactionRequest>,
    ) -> Result<Self> {
        let nebula_store = NebulaStore::new(device)?;
        Ok(CompactionAgent {
            rx,
            kvcache_mgr,
            nebula_store,
            load: 0.0,
            max_load: 0.8, // 80% max utilization
            control_plane_tx,
        })
    }

    pub async fn run(&mut self) -> Result<()> {
        while let Some(request) = self.rx.recv().await {
            if self.load < self.max_load {
                self.load += 0.1; // Simulate load increase
                
                match self.perform_compaction(&request).await {
                    Ok(_) => println!("Compaction job {} completed", request.job_id),
                    Err(e) => {
                        println!("Compaction failed for job {}: {}. Retrying...", request.job_id, e);
                        // Fallback to Nebula
                        self.nebula_store.trigger_compaction()?;
                        self.kvcache_mgr.perform_local_compaction().await;
                    }
                }
                
                self.load -= 0.1; // Simulate load decrease
            } else {
                // Offload to Nebula if overloaded
                self.nebula_store.trigger_compaction()?;
            }
            
            // Notify control plane of load
            if let Err(e) = self.control_plane_tx.send(CompactionRequest {
                job_id: "load_update".to_string(),
                data_access: vec![],
                priority: 0,
            }).await {
                println!("Failed to update control plane: {}", e);
            }
        }
        Ok(())
    }

    async fn perform_compaction(&self, request: &CompactionRequest) -> Result<()> {
        // Optimized compaction logic
        let device = self.nebula_store.get_device();
        let data = Tensor::ones((1024, 1024), DType::F32, &device)?;
        self.nebula_store.store_tensor("compaction_data", &data)?;
        
        // Perform remote compaction
        self.kvcache_mgr.perform_remote_compaction(
            &request.job_id, 
            &request.data_access
        ).await?;
        
        Ok(())
    }
}

