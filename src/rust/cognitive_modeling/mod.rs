use rand::Rng;
use std::collections::HashMap;

/// CognitiveModel represents a cognitive model that uses Bayesian inference
/// to update its theory space based on observations and likelihoods.
/// It maintains a theory space, evidence, and a prior probability.
/// /// The theory space is a mapping of theory IDs to their probabilities,
/// /// while evidence is a vector of observations and their likelihoods.
/// /// The model can add evidence, update the theory space based on the evidence,
/// /// and compute likelihoods for theories based on the evidence.
/// /// The model uses a simplified logic-based approach to compute likelihoods,


pub struct CognitiveModel {
    theory_space: HashMap<String, f64>, // Theory ID -> Probability
    evidence: Vec<(String, f64)>,      // (Observation, Likelihood)
    prior: f64,
}

impl CognitiveModel {
    pub fn new() -> Self {
        CognitiveModel {
            theory_space: HashMap::new(),
            evidence: Vec::new(),
            prior: 0.5, // Default prior probability
        }
    }

    pub fn add_evidence(&mut self, observation: String, likelihood: f64) {
        self.evidence.push((observation, likelihood));
    }

    pub fn update_theory_space(&mut self) {
        let mut total_prob = 0.0;
        for (theory_id, _) in self.theory_space.iter_mut() {
            let likelihood = self.compute_likelihood(theory_id);
            let posterior = self.prior * likelihood;
            *self.theory_space.get_mut(theory_id).unwrap() = posterior;
            total_prob += posterior;
        }
        // Normalize probabilities
        for (_, prob) in self.theory_space.iter_mut() {
            *prob /= total_prob;
        }
    }

    fn compute_likelihood(&self, theory_id: &str) -> f64 {
        let mut likelihood = 1.0;
        for (obs, _) in &self.evidence {
            // Placeholder: Logic-based likelihood computation
            if obs.contains(theory_id) {
                likelihood *= 0.9; // Simplified match
            } else {
                likelihood *= 0.1;
            }
        }
        likelihood
    }
}

/// MCMCSearch performs a Markov Chain Monte Carlo search to explore the theory space
/// and update the cognitive model's theories based on simulated annealing.
/// /// It uses a temperature parameter to control the exploration-exploitation trade-off.
/// /// The search iteratively proposes new theories, computes their likelihoods,
/// /// and accepts or rejects them based on an acceptance ratio.
pub struct MCMCSearch {
    model: CognitiveModel,
    temperature: f64,
    step_size: f64,
}

impl MCMCSearch {
    pub fn new(model: CognitiveModel) -> Self {
        MCMCSearch {
            model,
            temperature: 1.0, // Initial temperature for simulated annealing
            step_size: 0.1,
        }
    }

    pub fn search(&mut self, iterations: usize) {
        let mut rng = rand::thread_rng();
        let mut current_theory = String::from("initial_theory");
        self.model.theory_space.insert(current_theory.clone(), 1.0);

        for _ in 0..iterations {
            let new_theory = self.propose_new_theory(&current_theory);
            let current_likelihood = self.model.compute_likelihood(&current_theory);
            let new_likelihood = self.model.compute_likelihood(&new_theory);

            let acceptance_ratio = (new_likelihood / current_likelihood).exp() / self.temperature;
            if rng.gen::<f64>() < acceptance_ratio.min(1.0) {
                current_theory = new_theory;
                self.model.theory_space.insert(current_theory.clone(), new_likelihood);
            }

            self.model.update_theory_space();
            self.temperature *= 0.99; // Cool down
        }
    }

    fn propose_new_theory(&self, current: &str) -> String {
        let mut rng = rand::thread_rng();
        format!("{}_variant{}", current, rng.gen_range(0..100))
    }
}