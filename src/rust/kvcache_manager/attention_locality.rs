
// SPDX-License-Identifier: Apache-2.0
// This file is part of the Zeta Reticula - Fandango  project, which is licensed under the Apache License 2.0.

// This file contains functions to calculate entropy and measure locality of attention data.
// The entropy function calculates the uncertainty in the data distribution,
// while the locality function measures how close values are to the mean of the data.


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

