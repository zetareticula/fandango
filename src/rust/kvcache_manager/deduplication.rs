use bloomfilter::Bloom;

pub fn deduplicate_kvcache(data: &[f32]) -> Vec<f32> {
    const BLOCK_SIZE: usize = 64;
    const BLOOM_SIZE: usize = 10000;
    const FALSE_POSITIVE_RATE: f64 = 0.01;

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