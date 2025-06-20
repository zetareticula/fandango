
// SPDX-License-Identifier: Apache-2.0
// This file is part of the Zeta Reticula - Fandango  project, which is licensed under the Apache License 2.0.

// This file contains functions to calculate entropy and measure locality of attention data.
// The entropy function calculates the uncertainty in the data distribution,
// while the locality function measures how close values are to the mean of the data.

use std::f32;

// Calculate the entropy of a given data slice.
pub fn calculate_entropy(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }

    // Normalize the data to avoid division by zero
    let max_val = data.iter().cloned().fold(0.0, f32::max); // Find the maximum value
    let total = data.len() as f32; // Total number of elements

    if max_val == 0.0 {
        return 0.0; // If all values are zero, entropy is zero
    }

    // Normalize the data by dividing each value by the maximum value
    let data: Vec<f32> = data.iter().map(|&val| val / max_val).collect();

    // Recalculate the total after normalization
    let total: f32 = data.iter().map(|&val| val.abs()).sum();

    // Calculate the entropy
    let mut entropy = 0.0;
    for &val in &data {
        let prob = val.abs() / total;
        if prob > 0.0 {
            entropy -= prob * prob.log2();
        }
    }
    entropy / total.log2()
}

pub fn measure_locality(data: &[f32]) -> f32 {
    // Simplified locality measure (to be expanded)
    let window_size = 10;
    let mut locality = 0.0;
    for i in 0..data.len().saturating_sub(window_size) {
        let window = &data[i..i + window_size];
        locality += window.iter().sum::<f32>() / window.len() as f32;
    }
    locality / data.len() as f32
}

