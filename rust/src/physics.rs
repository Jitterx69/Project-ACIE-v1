//! Physics constraint evaluation

use ndarray::{Array1, ArrayView1};

/// Check physics constraints on latent variables
pub fn check_constraint(latent: Array1<f32>, constraint_type: &str) -> f32 {
    match constraint_type {
        "conservation" => check_conservation(&latent.view()),
        "virial" => check_virial(&latent.view()),
        "positivity" => check_positivity(&latent.view()),
        "bounds" => check_bounds(&latent.view()),
        _ => 0.0,
    }
}

/// Conservation law violation
fn check_conservation(latent: &ArrayView1<f32>) -> f32 {
    // Sum of first 100 components should be constant
    // Violation = |sum - expected|^2
    let sum: f32 = latent.slice(ndarray::s![0..100.min(latent.len())]).sum();
    let expected = 0.0;  // Normalized to zero
    (sum - expected).powi(2)
}

/// Virial theorem violation: 2K + U ≈ 0
fn check_virial(latent: &ArrayView1<f32>) -> f32 {
    if latent.len() < 200 {
        return 0.0;
    }
    
    // Kinetic energy proxy (first 100 components)
    let kinetic: f32 = latent
        .slice(ndarray::s![0..100])
        .iter()
        .map(|&x| x * x)
        .sum();
    
    // Potential energy proxy (next 100 components)
    let potential: f32 = latent.slice(ndarray::s![100..200]).sum();
    
    // Virial: 2K + U should be small
    let virial = 2.0 * kinetic + potential;
    virial.powi(2)
}

/// Positivity constraint: certain quantities must be > 0
fn check_positivity(latent: &ArrayView1<f32>) -> f32 {
    // First 500 components should be non-negative (mass, luminosity, etc.)
    let end = 500.min(latent.len());
    latent
        .slice(ndarray::s![0..end])
        .iter()
        .map(|&x| if x < 0.0 { x * x } else { 0.0 })
        .sum::<f32>()
}

/// Boundary constraints
fn check_bounds(latent: &ArrayView1<f32>) -> f32 {
    // All values should be in reasonable range [-10, 10]
    latent
        .iter()
        .map(|&x| {
            if x < -10.0 {
                (x + 10.0).powi(2)
            } else if x > 10.0 {
                (x - 10.0).powi(2)
            } else {
                0.0
            }
        })
        .sum()
}

/// Parallel batch constraint evaluation
pub fn batch_check_constraints(
    latents: &[Array1<f32>],
    constraint_type: &str,
) -> Vec<f32> {
    use rayon::prelude::*;
    
    latents
        .par_iter()
        .map(|l| check_constraint(l.clone(), constraint_type))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_conservation() {
        let latent = arr1(&[0.5; 100]);
        let violation = check_conservation(&latent.view());
        assert!(violation > 0.0);  // Non-zero sum violates conservation
    }

    #[test]
    fn test_positivity() {
        let mut latent_vec = vec![1.0; 500];
        latent_vec[0] = -1.0;  // Violation
        let latent = Array1::from_vec(latent_vec);
        let violation = check_positivity(&latent.view());
        assert!(violation > 0.0);
    }

    #[test]
    fn test_bounds() {
        let latent = arr1(&[15.0, 5.0, -15.0]);
        let violation = check_bounds(&latent.view());
        assert!(violation > 0.0);  // Values outside [-10, 10]
    }
}
