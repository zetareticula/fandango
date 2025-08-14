use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use candle_core::{DType, Device, Tensor};
use quantization_server::QuantizedModel;
use std::time::Duration;

fn quantization_benchmark(c: &mut Criterion) {
    let device = Device::Cpu;
    let sizes = [128, 256, 512, 1024];
    
    let mut group = c.benchmark_group("quantization");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(10);
    
    for &size in &sizes {
        let weights = Tensor::randn(0f32, 1.0, &[size, size], &device).unwrap();
        
        group.bench_with_input(BenchmarkId::new("quantize", size), &size, |b, _| {
            b.iter(|| {
                QuantizedModel::quantize(&weights, 8).unwrap();
            })
        });
        
        let quantized = QuantizedModel::quantize(&weights, 8).unwrap();
        let input = Tensor::randn(0f32, 1.0, &[1, size], &device).unwrap();
        
        group.bench_with_input(BenchmarkId::new("inference", size), &size, |b, _| {
            b.iter(|| {
                quantized.quantized_matmul(&input).unwrap();
            })
        });
    }
    
    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default().with_plots();
    targets = quantization_benchmark
);
criterion_main!(benches);
