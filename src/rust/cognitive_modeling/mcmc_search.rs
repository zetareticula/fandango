use crate::cognitive_modeling::CognitiveModel;
use rand::Rng;
use std::time::Instant;

pub struct MCMCSearch {
    model: CognitiveModel,
    temperature: f64,
    step_size: f64,
}

impl MCMCSearch {
    pub fn new() -> Self {
        MCMCSearch {
            model: CognitiveModel::new(),
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