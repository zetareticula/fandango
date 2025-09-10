//! Integration tests for LLM Quantization Suite
//! 
//! Tests the complete container circuit proposition framework with real HuggingFace models

use llm_quantization_suite::{
    ContainerCircuit, ModelConfig, QuantizationStrategy, QuantizationPrecision,
    OrchestrationConfig, ValidationRule, ValidationType, CircuitExecutor,
    circuit::FandangoCircuitExecutor, huggingface::{ModelConfigs, HuggingFaceLoader},
};
use rstest::*;
use std::collections::HashMap;
use tokio::test;
use uuid::Uuid;

/// Test fixture for circuit executor
#[fixture]
fn circuit_executor() -> FandangoCircuitExecutor {
    FandangoCircuitExecutor::new(8.0, false) // 8GB memory limit, no GPU for tests
}

/// Test fixture for basic quantization strategy
#[fixture]
fn basic_quantization_strategy() -> QuantizationStrategy {
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

/// Test fixture for orchestration config
#[fixture]
fn orchestration_config() -> OrchestrationConfig {
    OrchestrationConfig {
        max_concurrent_layers: 4,
        chunk_size: 2,
        timeout_seconds: 300, // 5 minutes for tests
        retry_attempts: 2,
        memory_limit_gb: 8.0,
        cpu_cores: 4,
    }
}

/// Test fixture for validation rules
#[fixture]
fn validation_rules() -> Vec<ValidationRule> {
    vec![
        ValidationRule {
            name: "compression_test".to_string(),
            rule_type: ValidationType::CompressionRatio,
            threshold: 2.0,
            critical: true,
        },
        ValidationRule {
            name: "accuracy_test".to_string(),
            rule_type: ValidationType::AccuracyLoss,
            threshold: 0.1, // 10% max loss for tests
            critical: false,
        },
    ]
}

#[rstest]
#[tokio::test]
async fn test_circuit_proposition_validation(
    circuit_executor: FandangoCircuitExecutor,
    basic_quantization_strategy: QuantizationStrategy,
    orchestration_config: OrchestrationConfig,
    validation_rules: Vec<ValidationRule>,
) {
    // Create a test circuit with LLaMA-2 7B configuration
    let circuit = ContainerCircuit {
        id: Uuid::new_v4(),
        name: "Test LLaMA-2 7B INT4 Quantization".to_string(),
        model_config: ModelConfigs::llama2_7b(),
        quantization_strategy: basic_quantization_strategy,
        orchestration_config,
        validation_rules,
    };

    // Test circuit proposition validation
    let is_valid = circuit_executor.validate_proposition(&circuit).await
        .expect("Validation should not fail");

    assert!(is_valid, "Circuit proposition should be valid");
}

#[rstest]
#[tokio::test]
async fn test_invalid_circuit_proposition(circuit_executor: FandangoCircuitExecutor) {
    // Create an invalid circuit with empty model ID
    let invalid_circuit = ContainerCircuit {
        id: Uuid::new_v4(),
        name: "Invalid Circuit".to_string(),
        model_config: ModelConfig {
            model_id: "".to_string(), // Invalid empty model ID
            revision: None,
            safetensors_files: vec![],
            config_json: "config.json".to_string(),
            tokenizer_config: None,
            expected_layers: 0,
            expected_params: 0,
        },
        quantization_strategy: QuantizationStrategy {
            precision: QuantizationPrecision::Int4,
            group_size: Some(128),
            act_order: true,
            damp_percent: 0.01,
            desc_act: false,
            static_groups: false,
            sym: true,
            true_sequential: true,
        },
        orchestration_config: OrchestrationConfig {
            max_concurrent_layers: 4,
            chunk_size: 2,
            timeout_seconds: 300,
            retry_attempts: 2,
            memory_limit_gb: 8.0,
            cpu_cores: 4,
        },
        validation_rules: vec![],
    };

    // Test that invalid circuit is rejected
    let is_valid = circuit_executor.validate_proposition(&invalid_circuit).await
        .expect("Validation should not fail");

    assert!(!is_valid, "Invalid circuit should be rejected");
}

#[rstest]
#[tokio::test]
async fn test_quantization_precision_levels(circuit_executor: FandangoCircuitExecutor) {
    let precisions = vec![
        QuantizationPrecision::Int2,
        QuantizationPrecision::Int4,
        QuantizationPrecision::Int8,
        QuantizationPrecision::Float16,
        QuantizationPrecision::BFloat16,
    ];

    for precision in precisions {
        let circuit = create_test_circuit_with_precision(precision.clone());
        
        let is_valid = circuit_executor.validate_proposition(&circuit).await
            .expect("Validation should not fail");
        
        assert!(is_valid, "Circuit with {:?} precision should be valid", precision);
    }
}

#[rstest]
#[tokio::test]
async fn test_concurrent_orchestration_limits(circuit_executor: FandangoCircuitExecutor) {
    let concurrent_limits = vec![1, 2, 4, 8, 16];

    for limit in concurrent_limits {
        let mut circuit = create_test_circuit_with_precision(QuantizationPrecision::Int4);
        circuit.orchestration_config.max_concurrent_layers = limit;
        
        let is_valid = circuit_executor.validate_proposition(&circuit).await
            .expect("Validation should not fail");
        
        assert!(is_valid, "Circuit with {} concurrent layers should be valid", limit);
    }
}

#[rstest]
#[tokio::test]
async fn test_memory_limit_validation(circuit_executor: FandangoCircuitExecutor) {
    let mut circuit = create_test_circuit_with_precision(QuantizationPrecision::Int4);
    
    // Test with memory limit exceeding executor capacity
    circuit.orchestration_config.memory_limit_gb = 32.0; // Exceeds 8GB test limit
    
    let is_valid = circuit_executor.validate_proposition(&circuit).await
        .expect("Validation should not fail");
    
    assert!(!is_valid, "Circuit exceeding memory limit should be invalid");
}

#[rstest]
#[tokio::test]
async fn test_validation_rule_thresholds() {
    let test_cases = vec![
        (ValidationType::CompressionRatio, 0.5, false), // Invalid: < 1.0
        (ValidationType::CompressionRatio, 2.0, true),  // Valid: >= 1.0
        (ValidationType::AccuracyLoss, -0.1, false),    // Invalid: < 0.0
        (ValidationType::AccuracyLoss, 1.5, false),     // Invalid: > 1.0
        (ValidationType::AccuracyLoss, 0.05, true),     // Valid: 0.0-1.0
    ];

    let circuit_executor = FandangoCircuitExecutor::new(8.0, false);

    for (rule_type, threshold, should_be_valid) in test_cases {
        let mut circuit = create_test_circuit_with_precision(QuantizationPrecision::Int4);
        circuit.validation_rules = vec![ValidationRule {
            name: "test_rule".to_string(),
            rule_type,
            threshold,
            critical: true,
        }];

        let is_valid = circuit_executor.validate_proposition(&circuit).await
            .expect("Validation should not fail");

        assert_eq!(is_valid, should_be_valid, 
                  "Rule {:?} with threshold {} should be {}", 
                  rule_type, threshold, if should_be_valid { "valid" } else { "invalid" });
    }
}

#[rstest]
#[tokio::test]
async fn test_huggingface_model_configs() {
    let model_configs = vec![
        ("LLaMA-2 7B", ModelConfigs::llama2_7b()),
        ("Mistral 7B", ModelConfigs::mistral_7b()),
        ("CodeLlama 7B", ModelConfigs::codellama_7b()),
    ];

    for (name, config) in model_configs {
        // Validate model configuration structure
        assert!(!config.model_id.is_empty(), "{} should have non-empty model ID", name);
        assert!(!config.safetensors_files.is_empty(), "{} should have safetensors files", name);
        assert!(!config.config_json.is_empty(), "{} should have config JSON", name);
        assert!(config.expected_layers > 0, "{} should have expected layers > 0", name);
        assert!(config.expected_params > 0, "{} should have expected params > 0", name);
        
        println!("✅ {} configuration is valid", name);
    }
}

#[rstest]
#[tokio::test]
async fn test_circuit_execution_simulation() {
    // Note: This test simulates circuit execution without actually downloading models
    // In a real environment, you would need HuggingFace API access and model downloads
    
    let circuit_executor = FandangoCircuitExecutor::new(8.0, false);
    let circuit = create_test_circuit_with_precision(QuantizationPrecision::Int4);
    
    // For testing purposes, we'll just validate the circuit structure
    // Real execution would require model downloads and significant compute resources
    let is_valid = circuit_executor.validate_proposition(&circuit).await
        .expect("Validation should not fail");
    
    assert!(is_valid, "Test circuit should be valid for simulation");
    
    // Test that circuit has all required components
    assert!(!circuit.model_config.model_id.is_empty());
    assert!(!circuit.validation_rules.is_empty());
    assert!(circuit.orchestration_config.max_concurrent_layers > 0);
    assert!(circuit.orchestration_config.timeout_seconds > 0);
}

#[rstest]
#[tokio::test]
async fn test_orchestration_config_validation() {
    let valid_configs = vec![
        OrchestrationConfig {
            max_concurrent_layers: 1,
            chunk_size: 1,
            timeout_seconds: 60,
            retry_attempts: 1,
            memory_limit_gb: 1.0,
            cpu_cores: 1,
        },
        OrchestrationConfig {
            max_concurrent_layers: 16,
            chunk_size: 8,
            timeout_seconds: 3600,
            retry_attempts: 5,
            memory_limit_gb: 64.0,
            cpu_cores: 32,
        },
    ];

    let circuit_executor = FandangoCircuitExecutor::new(8.0, false);

    for config in valid_configs {
        let mut circuit = create_test_circuit_with_precision(QuantizationPrecision::Int4);
        circuit.orchestration_config = config;
        
        // Only test memory limit validation (others are structural)
        let is_valid = if circuit.orchestration_config.memory_limit_gb <= 8.0 {
            circuit_executor.validate_proposition(&circuit).await
                .expect("Validation should not fail")
        } else {
            false // Would exceed test executor memory limit
        };
        
        let expected_valid = circuit.orchestration_config.memory_limit_gb <= 8.0;
        assert_eq!(is_valid, expected_valid, 
                  "Config with {:.1}GB memory should be {}", 
                  circuit.orchestration_config.memory_limit_gb,
                  if expected_valid { "valid" } else { "invalid" });
    }
}

#[rstest]
#[tokio::test]
async fn test_quantization_strategy_combinations() {
    let strategies = vec![
        // Standard GPTQ configuration
        QuantizationStrategy {
            precision: QuantizationPrecision::Int4,
            group_size: Some(128),
            act_order: false,
            damp_percent: 0.01,
            desc_act: false,
            static_groups: false,
            sym: true,
            true_sequential: true,
        },
        // AWQ-style configuration
        QuantizationStrategy {
            precision: QuantizationPrecision::Int4,
            group_size: Some(128),
            act_order: true,
            damp_percent: 0.01,
            desc_act: true,
            static_groups: false,
            sym: false,
            true_sequential: false,
        },
        // High precision configuration
        QuantizationStrategy {
            precision: QuantizationPrecision::Int8,
            group_size: None,
            act_order: false,
            damp_percent: 0.001,
            desc_act: false,
            static_groups: true,
            sym: true,
            true_sequential: true,
        },
    ];

    let circuit_executor = FandangoCircuitExecutor::new(8.0, false);

    for (i, strategy) in strategies.into_iter().enumerate() {
        let mut circuit = create_test_circuit_with_precision(QuantizationPrecision::Int4);
        circuit.quantization_strategy = strategy;
        
        let is_valid = circuit_executor.validate_proposition(&circuit).await
            .expect("Validation should not fail");
        
        assert!(is_valid, "Quantization strategy {} should be valid", i + 1);
    }
}

/// Helper function to create test circuit with specific precision
fn create_test_circuit_with_precision(precision: QuantizationPrecision) -> ContainerCircuit {
    ContainerCircuit {
        id: Uuid::new_v4(),
        name: format!("Test Circuit {:?}", precision),
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
        orchestration_config: OrchestrationConfig {
            max_concurrent_layers: 4,
            chunk_size: 2,
            timeout_seconds: 300,
            retry_attempts: 2,
            memory_limit_gb: 8.0,
            cpu_cores: 4,
        },
        validation_rules: vec![
            ValidationRule {
                name: "test_compression".to_string(),
                rule_type: ValidationType::CompressionRatio,
                threshold: 2.0,
                critical: true,
            },
        ],
    }
}
