use bloomfilter::Bloom;
use fxhash;
use std::vec::Vec;
use std::hash::Hash;
use std::hash::Hasher;
use std::collections::HashSet;
use std::iter::FromIterator;

/// A function to deduplicate a vector of f32 values using a Bloom filter.
/// This function processes the input data in blocks of a specified size,
/// tracking unique blocks to avoid duplicates.


/// Deduplicates a vector of f32 values using a Bloom filter to track unique blocks.
pub fn deduplicate_kvcache(data: &[f32]) -> Vec<f32> {
    const BLOCK_SIZE: usize = 64; // Size of each block to process
    const BLOOM_SIZE: usize = 10000; // Size of the Bloom filter
    const FALSE_POSITIVE_RATE: f64 = 0.01; // Desired false positive rate for the Bloom filter

    let mut bloom = Bloom::new(BLOOM_SIZE, FALSE_POSITIVE_RATE);
    let mut unique_blocks = Vec::new();

    for chunk in data.chunks(BLOCK_SIZE) {
        let hash = fxhash::hash64(chunk);
        if !bloom.check(&hash.to_be_bytes()) {
            bloom.set(&hash.to_be_bytes());
            unique_blocks.extend_from_slice(chunk);
        }
    }
    unique_blocks
}



