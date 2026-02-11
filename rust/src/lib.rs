//! ACIE Rust Core
//! 
//! High-performance components for the Astronomical Counterfactual Inference Engine.
//! 
//! Provides:
//! - Fast tensor operations
//! - Parallel data loading
//! - SCM graph algorithms
//! - Physics constraint evaluation

use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2};
use ndarray::{Array1, Array2};
use rayon::prelude::*;

pub mod tensor_ops;
pub mod scm_graph;
pub mod physics;
pub mod data_loader;

/// Fast matrix multiplication using optimized BLAS
#[pyfunction]
fn fast_matmul(
    py: Python,
    a: &PyArray2<f32>,
    b: &PyArray2<f32>,
) -> PyResult<Py<PyArray2<f32>>> {
    let a_array = unsafe { a.as_array() };
    let b_array = unsafe { b.as_array() };
    
    let result = a_array.dot(&b_array);
    
    Ok(PyArray2::from_owned_array(py, result).to_owned())
}

/// Parallel physics constraint evaluation
#[pyfunction]
fn evaluate_physics_constraints(
    py: Python,
    latent: &PyArray2<f32>,
    constraint_type: &str,
) -> PyResult<Py<PyArray1<f32>>> {
    let latent_array = unsafe { latent.as_array() };
    
    let violations: Vec<f32> = latent_array
        .axis_iter(ndarray::Axis(0))
        .into_par_iter()
        .map(|row| {
            physics::check_constraint(row.to_owned(), constraint_type)
        })
        .collect();
    
    let violations_array = Array1::from_vec(violations);
    Ok(PyArray1::from_owned_array(py, violations_array).to_owned())
}

/// Fast topological sort for SCM
#[pyfunction]
fn topological_sort(edges: Vec<(usize, usize)>, num_nodes: usize) -> PyResult<Vec<usize>> {
    scm_graph::topological_sort_fast(edges, num_nodes)
}

/// Parallel CSV chunk loading
#[pyfunction]
fn load_csv_parallel(
    path: &str,
    chunk_size: usize,
    max_rows: Option<usize>,
) -> PyResult<Vec<Vec<f32>>> {
    data_loader::load_csv_parallel(path, chunk_size, max_rows)
}

/// Python module definition
#[pymodule]
fn acie_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fast_matmul, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_physics_constraints, m)?)?;
    m.add_function(wrap_pyfunction!(topological_sort, m)?)?;
    m.add_function(wrap_pyfunction!(load_csv_parallel, m)?)?;
    Ok(())
}
