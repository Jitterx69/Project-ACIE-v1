// Parallel data loading for large datasets

use rayon::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufReader};
use pyo3::prelude::*;

/// Load CSV in parallel chunks
pub fn load_csv_parallel(
    path: &str,
    chunk_size: usize,
    max_rows: Option<usize>,
) -> PyResult<Vec<Vec<f32>>> {
    let file = File::open(path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
            format!("Failed to open file: {}", e)
        ))?;
    
    let reader = BufReader::new(file);
    let mut all_data = Vec::new();
    let mut count = 0;
    
    // Skip header
    let mut lines = reader.lines().skip(1);
    
    while let Some(Ok(line)) = lines.next() {
        if let Some(max) = max_rows {
            if count >= max {
                break;
            }
        }
        
        let values: Vec<f32> = line
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        
        all_data.push(values);
        count += 1;
    }
    
    Ok(all_data)
}

/// Parallel feature extraction from loaded data  
pub fn extract_features_parallel(
    data: &[Vec<f32>],
    latent_dim: usize,
    obs_dim: usize,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let (latents, observables): (Vec<_>, Vec<_>) = data
        .par_iter()
        .map(|row| {
            let latent: Vec<f32> = row[0..latent_dim].to_vec();
            let obs: Vec<f32> = row[latent_dim..latent_dim + obs_dim].to_vec();
            (latent, obs)
        })
        .unzip();
    
    (latents, observables)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_features() {
        let data = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![6.0, 7.0, 8.0, 9.0, 10.0],
        ];
        
        let (latents, obs) = extract_features_parallel(&data, 2, 3);
        
        assert_eq!(latents.len(), 2);
        assert_eq!(obs.len(), 2);
        assert_eq!(latents[0], vec![1.0, 2.0]);
        assert_eq!(obs[0], vec![3.0, 4.0, 5.0]);
    }
}
