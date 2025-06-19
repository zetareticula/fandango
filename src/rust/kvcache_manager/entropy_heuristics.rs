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

