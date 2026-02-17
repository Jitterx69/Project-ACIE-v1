//! High-performance tensor operations

use ndarray::{Array1, Array2, ArrayView1};
use rayon::prelude::*;

/// Optimized batch matrix-vector multiplication
pub fn batch_matvec(matrices: &[Array2<f32>], vector: &Array1<f32>) -> Vec<Array1<f32>> {
    // Convert to standard implementation to avoid trait bound issues with AxisIter
    let matrices_vec: Vec<_> = matrices.iter().collect();
    matrices_vec.par_iter().map(|mat| mat.dot(vector)).collect()
}

/// Fast element-wise operations with SIMD
pub fn elementwise_tanh(input: &Array2<f32>) -> Array2<f32> {
    input.mapv(|x| x.tanh())
}

/// Vectorized ReLU
pub fn relu(input: &Array2<f32>) -> Array2<f32> {
    input.mapv(|x| x.max(0.0))
}

/// Compute variance along axis efficiently
pub fn variance_axis0(data: &Array2<f32>) -> Array1<f32> {
    let mean = data.mean_axis(ndarray::Axis(0)).unwrap();

    // Collect rows first to enable parallel iteration
    let rows: Vec<_> = data.axis_iter(ndarray::Axis(0)).collect();

    rows.into_par_iter()
        .map(|row| {
            row.iter()
                .zip(mean.iter())
                .map(|(x, m)| (x - m).powi(2))
                .sum::<f32>()
        })
        .collect::<Vec<f32>>()
        .into()
}

/// Fast cosine similarity between vectors
pub fn cosine_similarity(a: ArrayView1<f32>, b: ArrayView1<f32>) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    dot_product / (norm_a * norm_b + 1e-8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_relu() {
        let input = Array2::from_shape_vec((2, 3), vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0]).unwrap();
        let output = relu(&input);
        assert_eq!(output[[0, 0]], 0.0);
        assert_eq!(output[[0, 1]], 2.0);
        assert_eq!(output[[1, 0]], 4.0);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = arr1(&[1.0, 2.0, 3.0]);
        let b = arr1(&[1.0, 2.0, 3.0]);
        let sim = cosine_similarity(a.view(), b.view());
        assert!((sim - 1.0).abs() < 1e-6);
    }
}
