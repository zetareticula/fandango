//! Performance benchmarks for LLM quantization suite

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use llm_quantization_suite::{
    ContainerCircuit, QuantizationStrategy, QuantizationPrecision,
    OrchestrationConfig, ValidationRule, ValidationType,
    circuit::FandangoCircuitExecutor, huggingface::ModelConfigs,
};
use tokio::runtime::Runtime;
use uuid::Uuid;

fn benchmark_circuit_validation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let executor = FandangoCircuitExecutor::new(16.0, false);
    
    let precisions = vec![
        QuantizationPrecision::Int2,
        QuantizationPrecision::Int4,
        QuantizationPrecision::Int8,
        QuantizationPrecision::Float16,
    ];
    
    let mut group = c.benchmark_group("circuit_validation");
    
    for precision in precisions {
        let circuit = create_benchmark_circuit(precision.clone());
        
        group.bench_with_input(
            BenchmarkId::new("validate_proposition", format!("{:?}", precision)),
            &circuit,
            |b, circuit| {
                b.to_async(&rt).iter(|| async {
                    black_box(executor.validate_proposition(circuit).await.unwrap())
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_concurrent_orchestration(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let executor = FandangoCircuitExecutor::new(16.0, false);
    
    let concurrent_levels = vec![1, 2, 4, 8, 16];
    
    let mut group = c.benchmark_group("concurrent_orchestration");
    
    for concurrent in concurrent_levels {
        let mut circuit = create_benchmark_circuit(QuantizationPrecision::Int4);
        circuit.orchestration_config.max_concurrent_layers = concurrent;
        
        group.bench_with_input(
            BenchmarkId::new("validate_concurrent", concurrent),
            &circuit,
            |b, circuit| {
                b.to_async(&rt).iter(|| async {
                    black_box(executor.validate_proposition(circuit).await.unwrap())
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_model_configurations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let executor = FandangoCircuitExecutor::new(16.0, false);
    
    let models = vec![
        ("llama2_7b", ModelConfigs::llama2_7b()),
        ("mistral_7b", ModelConfigs::mistral_7b()),
        ("codellama_7b", ModelConfigs::codellama_7b()),
    ];
    
    let mut group = c.benchmark_group("model_configurations");
    
    for (name, model_config) in models {
        let circuit = ContainerCircuit {
            id: Uuid::new_v4(),
            name: format!("Benchmark {}", name),
            model_config,
            quantization_strategy: create_benchmark_strategy(),
            orchestration_config: create_benchmark_orchestration(),
            validation_rules: create_benchmark_validation_rules(),
        };
        
        group.bench_with_input(
            BenchmarkId::new("validate_model", name),
            &circuit,
            |b, circuit| {
                b.to_async(&rt).iter(|| async {
                    black_box(executor.validate_proposition(circuit).await.unwrap())
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_validation_rules(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let executor = FandangoCircuitExecutor::new(16.0, false);
    
    let rule_counts = vec![1, 5, 10, 20];
    
    let mut group = c.benchmark_group("validation_rules");
    
    for count in rule_counts {
        let mut circuit = create_benchmark_circuit(QuantizationPrecision::Int4);
        circuit.validation_rules = create_multiple_validation_rules(count);
        
        group.bench_with_input(
            BenchmarkId::new("validate_rules", count),
            &circuit,
            |b, circuit| {
                b.to_async(&rt).iter(|| async {
                    black_box(executor.validate_proposition(circuit).await.unwrap())
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_quantization_strategies(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let executor = FandangoCircuitExecutor::new(16.0, false);
    
    let group_sizes = vec![32, 64, 128, 256, 512];
    
    let mut group = c.benchmark_group("quantization_strategies");
    
    for group_size in group_sizes {
        let mut circuit = create_benchmark_circuit(QuantizationPrecision::Int4);
        circuit.quantization_strategy.group_size = Some(group_size);
        
        group.bench_with_input(
            BenchmarkId::new("validate_group_size", group_size),
            &circuit,
            |b, circuit| {
                b.to_async(&rt).iter(|| async {
                    black_box(executor.validate_proposition(circuit).await.unwrap())
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_memory_scaling(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let memory_limits = vec![4.0, 8.0, 16.0, 32.0, 64.0];
    
    let mut group = c.benchmark_group("memory_scaling");
    
    for memory_gb in memory_limits {
        let executor = FandangoCircuitExecutor::new(memory_gb, false);
        let mut circuit = create_benchmark_circuit(QuantizationPrecision::Int4);
        circuit.orchestration_config.memory_limit_gb = memory_gb * 0.8; // Use 80% of available
        
        group.bench_with_input(
            BenchmarkId::new("validate_memory", format!("{}GB", memory_gb)),
            &circuit,
            |b, circuit| {
                b.to_async(&rt).iter(|| async {
                    black_box(executor.validate_proposition(circuit).await.unwrap())
                });
            },
        );
    }
    
    group.finish();
}

// Helper functions

fn create_benchmark_circuit(precision: QuantizationPrecision) -> ContainerCircuit {
    ContainerCircuit {
        id: Uuid::new_v4(),
        name: format!("Benchmark Circuit {:?}", precision),
        model_config: ModelConfigs::llama2_7b(),
        quantization_strategy: QuantizationStrategy {
            precision,
            group_size: Some(128),
            act_order: true,
            damp_percent: 0.01,
            desc_act: false,
            static_groups: false,
            sym: true,
            true_sequential: true,
        },
        orchestration_config: create_benchmark_orchestration(),
        validation_rules: create_benchmark_validation_rules(),
    }
}

fn create_benchmark_strategy() -> QuantizationStrategy {
    QuantizationStrategy {
        precision: QuantizationPrecision::Int4,
        group_size: Some(128),
        act_order: true,
        damp_percent: 0.01,
        desc_act: false,
        static_groups: false,
        sym: true,
        true_sequential: true,
    }
}

fn create_benchmark_orchestration() -> OrchestrationConfig {
    OrchestrationConfig {
        max_concurrent_layers: 8,
        chunk_size: 4,
        timeout_seconds: 3600,
        retry_attempts: 3,
        memory_limit_gb: 16.0,
        cpu_cores: 8,
    }
}

fn create_benchmark_validation_rules() -> Vec<ValidationRule> {
    vec![
        ValidationRule {
            name: "compression_benchmark".to_string(),
            rule_type: ValidationType::CompressionRatio,
            threshold: 2.0,
            critical: true,
        },
        ValidationRule {
            name: "accuracy_benchmark".to_string(),
            rule_type: ValidationType::AccuracyLoss,
            threshold: 0.05,
            critical: true,
        },
        ValidationRule {
            name: "latency_benchmark".to_string(),
            rule_type: ValidationType::LatencyIncrease,
            threshold: 0.2,
            critical: false,
        },
    ]
}

fn create_multiple_validation_rules(count: usize) -> Vec<ValidationRule> {
    let mut rules = Vec::new();
    
    for i in 0..count {
        let rule_type = match i % 5 {
            0 => ValidationType::CompressionRatio,
            1 => ValidationType::AccuracyLoss,
            2 => ValidationType::LatencyIncrease,
            3 => ValidationType::MemoryReduction,
            4 => ValidationType::ThroughputMaintenance,
            _ => ValidationType::CompressionRatio,
        };
        
        let threshold = match rule_type {
            ValidationType::CompressionRatio => 2.0,
            ValidationType::AccuracyLoss => 0.05,
            ValidationType::LatencyIncrease => 0.2,
            ValidationType::MemoryReduction => 0.5,
            ValidationType::ThroughputMaintenance => 0.8,
        };
        
        rules.push(ValidationRule {
            name: format!("rule_{}", i),
            rule_type,
            threshold,
            critical: i < count / 2, // First half are critical
        });
    }
    
    rules
}

criterion_group!(
    benches,
    benchmark_circuit_validation,
    benchmark_concurrent_orchestration,
    benchmark_model_configurations,
    benchmark_validation_rules,
    benchmark_quantization_strategies,
    benchmark_memory_scaling
);

criterion_main!(benches);
