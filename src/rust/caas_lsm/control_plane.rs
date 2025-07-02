use std::collections::HashMap;
use tokio::sync::mpsc;
use candle_core::Device;
use anyhow::Result;

use crate::{
    nebula_integration::nebula_store::NebulaStore,
    kvcache_manager::KVCacheManager,
};

#[derive(Debug)]
pub struct CompactionRequest {
    pub job_id: String,
    pub data_access: Vec<String>,
    pub priority: i32,
}

struct ResourceMonitor {
    available_csas: Vec<String>,
    load_levels: HashMap<String, f32>,
}

pub struct ControlPlane {
    csas: HashMap<String, mpsc::Sender<CompactionRequest>>,
    kvcache_mgr: KVCacheManager,
    resource_monitor: ResourceMonitor,
    nebula_store: NebulaStore,
}

impl ControlPlane {
    pub fn new(kvcache_mgr: KVCacheManager, device: Device) -> Result<Self> {
        let nebula_store = NebulaStore::new(device)?;
        let (tx, rx) = mpsc::channel(32);
        let mut csas = HashMap::new();
        csas.insert("csa1".to_string(), tx);
        Ok(ControlPlane {
            csas,
            kvcache_mgr,
            resource_monitor: ResourceMonitor {
                available_csas: vec!["csa1".to_string()],
                load_levels: HashMap::new(),
            },
            nebula_store,
        })
    }

    pub async fn schedule_compaction(&mut self, request: CompactionRequest) {
        let optimal_csa = self.resource_monitor.available_csas
            .iter()
            .min_by_key(|&csa| self.resource_monitor.load_levels.get(csa).unwrap_or(&0.0))
            .cloned();

        if let Some(csa) = optimal_csa {
            if let Some(tx) = self.csas.get(&csa) {
                tx.send(request).await.unwrap();
            } else {
                self.nebula_store.trigger_compaction(); // Fallback to Nebula compaction
                self.kvcache_mgr.perform_local_compaction().await;
            }
        }
    }

    pub fn update_load(&mut self, csa: &str, load: f32) {
        self.resource_monitor.load_levels.insert(csa.to_string(), load);
    }
}