
// SPDX-License-Identifier: Apache-2.0
// This file is part of the Zeta Reticula - Fandango  project, which is licensed under the Apache License 2.0.

// This file contains functions to calculate entropy and measure locality of attention data.
// The entropy function calculates the uncertainty in the data distribution,
// while the locality function measures how close values are to the mean of the data.


//! Key-Value Cache Manager with Learned Structure Optimization
//!
//! This module provides a key-value cache that can optimize its storage layout
//! using learned structures based on access patterns.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant, SystemTime};

use candle_core::{Device, Tensor};
use hashbrown::{HashMap, hash_map::Entry};
use log;
use thiserror::Error;
use crate::storage_engine::StorageEngineError;

use crate::storage_engine::learned_structures::LearnedStructure;

/// Error type for KVCache operations
#[derive(Error, Debug)]
pub enum KVCacheError {
    #[error("Compaction error: {0}")]
    CompactionError(String),
    
    #[error("Update error: {0}")]
    UpdateError(String),
    
    #[error("Storage engine error: {0}")]
    StorageEngine(#[from] StorageEngineError),
    
    #[error(transparent)]
    CandleError(#[from] candle_core::Error),
}

/// Result type for KVCache operations
pub type Result<T> = std::result::Result<T, KVCacheError>;

/// Metrics for monitoring cache performance
#[derive(Debug)]
pub struct CacheMetrics {
    /// Number of cache hits
    hits: AtomicUsize,
    /// Number of cache misses
    misses: AtomicUsize,
    /// Total number of bytes stored
    total_bytes: AtomicUsize,
    /// Total number of operations performed
    operations: AtomicUsize,
    /// Total time spent in compaction (in microseconds)
    compaction_time: AtomicUsize,
    /// Number of compactions performed
    compactions: AtomicUsize,
    /// Timestamp of the last compaction
    last_compaction: parking_lot::Mutex<Option<SystemTime>>,
}

impl Default for CacheMetrics {
    fn default() -> Self {
        Self {
            hits: AtomicUsize::new(0),
            misses: AtomicUsize::new(0),
            total_bytes: AtomicUsize::new(0),
            operations: AtomicUsize::new(0),
            compaction_time: AtomicUsize::new(0),
            compactions: AtomicUsize::new(0),
            last_compaction: parking_lot::Mutex::new(None),
        }
    }
}

impl CacheMetrics {
    /// Records a cache hit
    pub fn record_hit(&self) {
        self.hits.fetch_add(1, Ordering::Relaxed);
        self.operations.fetch_add(1, Ordering::Relaxed);
    }

    /// Records a cache miss
    pub fn record_miss(&self) {
        self.misses.fetch_add(1, Ordering::Relaxed);
        self.operations.fetch_add(1, Ordering::Relaxed);
    }

    /// Records bytes stored
    pub fn record_bytes(&self, bytes: usize) {
        self.total_bytes.fetch_add(bytes, Ordering::Relaxed);
    }
    
    /// Records compaction time
    pub fn record_compaction(&self, duration: Duration) {
        self.compactions.fetch_add(1, Ordering::Relaxed);
        self.compaction_time.fetch_add(
            duration.as_micros() as usize, 
            Ordering::Relaxed
        );
        *self.last_compaction.lock() = Some(SystemTime::now());
    }

    /// Gets the current hit rate
    pub fn hit_rate(&self) -> f64 {
        let hits = self.hits.load(Ordering::Relaxed) as f64;
        let ops = self.operations.load(Ordering::Relaxed) as f64;
        if ops > 0.0 { hits / ops } else { 0.0 }
    }
    
    /// Gets the average compaction time in milliseconds
    pub fn avg_compaction_time_ms(&self) -> f64 {
        let total = self.compaction_time.load(Ordering::Relaxed) as f64;
        let count = self.compactions.load(Ordering::Relaxed) as f64;
        if count > 0.0 { total / 1000.0 / count } else { 0.0 }
    }
    
    /// Gets the time since last compaction
    pub fn time_since_last_compaction(&self) -> Option<Duration> {
        self.last_compaction.lock()
            .and_then(|t| t.elapsed().ok())
    }
}

/// Manages a key-value cache with learned structure optimization
pub struct KVCacheManager {
    /// In-memory key-value store
    cache: HashMap<Vec<u8>, Vec<u8>>,
    
    /// Learned structure for optimization
    learned_struct: Option<LearnedStructure>,
    
    /// Device for tensor operations
    device: Device,
    
    /// Write buffer
    buffer: Vec<u8>,
    
    /// Maximum buffer capacity in bytes
    buffer_capacity: usize,
    
    /// Performance metrics
    metrics: CacheMetrics,
    
    /// Time when the cache was created
    created_at: Instant,
}

impl KVCacheManager {
    /// Creates a new KVCacheManager with default settings
    /// 
    /// # Arguments
    /// * `device` - The device to use for tensor operations
    /// 
    /// # Returns
    /// A new instance of KVCacheManager with 1MB buffer capacity
    pub fn new_default(device: Device) -> Self {
        Self::new(device, 1024 * 1024) // 1MB default buffer
    }
    /// Creates a new KVCacheManager with the specified parameters
    ///
    /// # Arguments
    /// * `device` - The device to use for tensor operations
    /// * `buffer_capacity` - Maximum buffer capacity in bytes before compaction
    ///
    /// # Returns
    /// A new instance of KVCacheManager
    pub fn new(device: Device, buffer_capacity: usize) -> Self {
        Self {
            cache: HashMap::new(),
            learned_struct: None,
            device,
            buffer: Vec::with_capacity(buffer_capacity),
            buffer_capacity,
            metrics: CacheMetrics::default(),
            created_at: Instant::now(),
        }
    }
    
    /// Inserts a key-value pair into the cache
    /// 
    /// # Arguments
    /// * `key` - The key to insert
    /// * `value` - The value to insert
    /// 
    /// # Returns
    /// `Result<Option<Vec<u8>>>` - The previous value if the key existed
    pub fn insert(&mut self, key: Vec<u8>, value: Vec<u8>) -> Result<Option<Vec<u8>>> {
        self.metrics.record_bytes(value.len());
        
        // Check if we need to perform compaction
        if self.buffer.len() + value.len() > self.buffer_capacity {
            self.perform_local_compaction()?;
        }
        
        // Add to buffer and insert into cache
        self.buffer.extend_from_slice(&value);
        
        match self.cache.entry(key) {
            Entry::Occupied(mut entry) => {
                let old_value = entry.insert(value);
                Ok(Some(old_value))
            }
            Entry::Vacant(entry) => {
                entry.insert(value);
                Ok(None)
            }
        }
    }
    
    /// Inserts multiple key-value pairs in a batch
    /// 
    /// # Arguments
    /// * `items` - An iterator of (key, value) pairs to insert
    /// 
    /// # Returns
    /// `Result<usize>` - Number of items inserted
    pub fn insert_batch<I, K, V>(&mut self, items: I) -> Result<usize>
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<Vec<u8>>,
        V: Into<Vec<u8>>,
    {
        let mut count = 0;
        for (key, value) in items.into_iter() {
            let key = key.into();
            let value = value.into();
            self.insert(key, value)?;
            count += 1;
            
            // Check if we need to perform compaction periodically
            if count % 100 == 0 && self.buffer.len() > self.buffer_capacity / 2 {
                self.perform_local_compaction()?;
            }
        }
        
        // Final compaction if needed
        if !self.buffer.is_empty() {
            self.perform_local_compaction()?;
        }
        
        Ok(count)
    }
    
    /// Gets a value from the cache by key
    /// 
    /// # Arguments
    /// * `key` - The key to look up
    /// 
    /// # Returns
    /// `Option<&[u8]>` - The value if found, or None if not found
    pub fn get(&self, key: &[u8]) -> Option<&[u8]> {
        match self.cache.get(key) {
            Some(value) => {
                self.metrics.record_hit();
                Some(value.as_slice())
            }
            None => {
                self.metrics.record_miss();
                None
            }
        }
    }
    
    /// Removes a key-value pair from the cache
    /// 
    /// # Arguments
    /// * `key` - The key to remove
    /// 
    /// # Returns
    /// `Option<Vec<u8>>` - The removed value if it existed
    pub fn remove(&mut self, key: &[u8]) -> Option<Vec<u8>> {
        self.cache.remove(key)
    }
    
    /// Checks if the cache contains a key
    /// 
    /// # Arguments
    /// * `key` - The key to check
    /// 
    /// # Returns
    /// `bool` - True if the key exists in the cache
    pub fn contains_key(&self, key: &[u8]) -> bool {
        self.cache.contains_key(key)
    }
    
    /// Returns the number of key-value pairs in the cache
    /// 
    /// # Returns
    /// `usize` - The number of entries in the cache
    pub fn len(&self) -> usize {
        self.cache.len()
    }
    
    /// Checks if the cache is empty
    /// 
    /// # Returns
    /// `bool` - True if the cache contains no elements
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }
    
    /// Returns the current cache metrics
    /// 
    /// # Returns
    /// `&CacheMetrics` - Reference to the cache metrics
    pub fn metrics(&self) -> &CacheMetrics {
        &self.metrics
    }
    
    /// Performs local compaction of the key-value cache
    /// 
    /// This method optimizes the learned structure based on the current workload
    /// and adjusts the levels accordingly.
    /// 
    /// # Returns
    /// `Result<()>` - Ok on success, or an error if compaction fails
    pub fn perform_local_compaction(&mut self) -> Result<()> {
        let start_time = Instant::now();
        log::info!("Starting local compaction");
        
        // Skip if buffer is empty
        if self.buffer.is_empty() {
            log::info!("Skipping compaction: buffer is empty");
            return Ok(());
        }
        
        // Create a new learned structure if it doesn't exist
        if self.learned_struct.is_none() {
            log::info!("Initializing new learned structure");
            self.learned_struct = Some(LearnedStructure::new(self.device.clone(), 3)?);
        }
        
        let result = if let Some(ls) = &mut self.learned_struct {
            // Convert buffer to f32 values for optimization
            let buffer_f32: Vec<f32> = self.buffer.iter().map(|&b| b as f32 / 255.0).collect();
            
            // Create a 1D tensor from the buffer with explicit shape
            let tensor_1d = Tensor::from_slice(
                &buffer_f32,
                (buffer_f32.len(),),  // Note the trailing comma to make it a single-element tuple
                &self.device
            )?;
            
            // Reshape to 2D tensor with a single row
            let tensor = tensor_1d.reshape((1, buffer_f32.len()))?;
            
            // Optimize with 70% read workload assumption
            ls.optimize(&tensor, 0.7)?;
            
            // Clear the buffer after successful compaction
            self.buffer.clear();
            
            log::info!("Local compaction completed successfully");
            Ok(())
        } else {
            let err_msg = "Failed to initialize or access learned structure";
            log::error!("{}", err_msg);
            Err(KVCacheError::CompactionError(err_msg.to_string()))
        };
        
        // Record metrics
        let duration = start_time.elapsed();
        self.metrics.record_compaction(duration);
        
        result
    }
    
    /// Clears the cache and resets all metrics
    pub fn clear(&mut self) {
        self.cache.clear();
        self.buffer.clear();
        self.learned_struct = None;
        // Reset metrics
        self.metrics = CacheMetrics::default();
    }
    
    /// Returns the current buffer size in bytes
    pub fn buffer_size(&self) -> usize {
        self.buffer.len()
    }
    
    /// Returns the buffer capacity in bytes
    pub fn buffer_capacity(&self) -> usize {
        self.buffer_capacity
    }
    
    /// Returns the time since the cache was created
    pub fn uptime(&self) -> Duration {
        self.created_at.elapsed()
    }
}

impl Drop for KVCacheManager {
    fn drop(&mut self) {
        // Perform final compaction before dropping
        if !self.buffer.is_empty() {
            if let Err(e) = self.perform_local_compaction() {
                log::error!("Error during final compaction: {}", e);
            }
        }
        
        log::info!(
            "Dropping KVCacheManager after {:?} with {} items",
            self.uptime(),
            self.len()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};
    
    #[test]
    fn test_cache_basic_operations() {
        let device = Device::Cpu;
        let mut cache = KVCacheManager::new(device, 1024);
        
        // Test insert and get
        cache.insert(b"key1".to_vec(), b"value1".to_vec()).unwrap();
        assert_eq!(cache.get(b"key1"), Some(b"value1".as_ref()));
        
        // Test overwrite
        let old = cache.insert(b"key1".to_vec(), b"new_value".to_vec()).unwrap();
        assert_eq!(old, Some(b"value1".to_vec()));
        assert_eq!(cache.get(b"key1"), Some(b"new_value".as_ref()));
        
        // Test non-existent key
        assert_eq!(cache.get(b"nonexistent"), None);
        
        // Test contains_key
        assert!(cache.contains_key(b"key1"));
        assert!(!cache.contains_key(b"nonexistent"));
        
        // Test remove
        let removed = cache.remove(b"key1").unwrap();
        assert_eq!(removed, b"new_value");
        assert!(!cache.contains_key(b"key1"));
        
        // Test len and is_empty
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }
    
    #[test]
    fn test_batch_operations() {
        let device = Device::Cpu;
        let mut cache = KVCacheManager::new(device, 1024);
        
        // Prepare test data
        let items = vec![
            (b"key1".to_vec(), b"value1".to_vec()),
            (b"key2".to_vec(), b"value2".to_vec()),
            (b"key3".to_vec(), b"value3".to_vec()),
        ];
        
        // Test batch insert
        let count = cache.insert_batch(items).unwrap();
        assert_eq!(count, 3);
        assert_eq!(cache.len(), 3);
        
        // Test metrics
        let metrics = cache.metrics();
        assert_eq!(metrics.hits.load(Ordering::Relaxed), 0);
        assert!(metrics.misses.load(Ordering::Relaxed) >= 0);
        
        // Test clear
        cache.clear();
        assert!(cache.is_empty());
    }
}