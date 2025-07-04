// This file is part of the Zeta Reticula - Fandango project, which is released under the Apache License 2.0.
// SPDX-License-Identifier: Apache-2.0

// This file contains functions to calculate entropy and measure locality of attention data.
use std::f32;
use std::iter::Sum;
use std::ops::{Add, Div, Mul};

// Calculate the mean of a given data slice.
pub fn calculate_mean(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f32>() / data.len() as f32
}


//calculate entropy and measure locality of attention data
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
    // Iterate through the data and calculate the probability of each value
    for &val in data {
        let prob = val.abs() / total;
        if prob > 0.0 {
            entropy -= prob * prob.log2();
        }
    }
    entropy / total.log2()
}

// Measure locality of attention data based on how close values are to the mean
pub fn measure_locality(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }

    // Calculate the mean of the data
    let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;

    // Calculate the locality based on how close values are to the mean
    let locality: f32 = data.iter()
        .map(|&val| (val - mean).abs())
        .sum::<f32>() / data.len() as f32;

    // Normalize locality to a range of 0.0 to 1.0
    locality / mean.abs().max(1.0)
}



// Calculate variance of the data
pub fn calculate_variance(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let mean = calculate_mean(data);
    data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32
}



