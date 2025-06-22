
// SPDX-License-Identifier: Apache-2.0
// This file is part of the Zeta Reticula - Fandango  project, which is licensed under the Apache License 2.0.

// This file contains functions to calculate entropy and measure locality of attention data.
// The entropy function calculates the uncertainty in the data distribution,
// while the locality function measures how close values are to the mean of the data.

use std::f32;
use std::iter::Sum;
use std::ops::Add;
use std::ops::Div;
use std::ops::Mul;



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

// Additional utility functions can be added here for more complex locality measures
pub fn calculate_mean(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f32>() / data.len() as f32
}

pub fn calculate_variance(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let mean = calculate_mean(data);
    data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32
}

// Calculate the standard deviation of the data
pub fn calculate_std_dev(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    calculate_variance(data).sqrt()
}

// Calculate the locality of the data based on its mean and standard deviation
pub fn calculate_locality(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let mean = calculate_mean(data);
    let std_dev = calculate_std_dev(data);
    
    // Locality is defined as the inverse of the standard deviation
    // This is a simplified measure; more complex locality measures can be implemented
    if std_dev == 0.0 {
        return 1.0; // If all values are the same, locality is maximum
    }
    1.0 / std_dev
}

// Additional functions can be added to enhance the locality measure

#[allow(dead_code)]
pub fn calculate_entropy_and_locality(data: &[f32]) -> (f32, f32) {
    let entropy = calculate_entropy(data);
    let locality = measure_locality(data);
    (entropy, locality)
}

#[allow(dead_code)]
pub fn calculate_entropy_and_locality_with_mean(data: &[f32]) -> (f32, f32, f32) {
    let entropy = calculate_entropy(data);
    let locality = measure_locality(data);
    let mean = calculate_mean(data);
    (entropy, locality, mean)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_entropy() {
        let data = vec![1.0, 2.0, 3.0];
        let entropy = calculate_entropy(&data);
        assert!(entropy > 0.0);
    }

    #[test]
    fn test_measure_locality() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let locality = measure_locality(&data);
        assert!(locality > 0.0);
    }
}
