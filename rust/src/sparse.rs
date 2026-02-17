//! Sparse Matrix Operations
//!
//! Implements Compressed Sparse Row (CSR) format and fast Sparse-Dense Matrix Multiplication (SpMM).

use ndarray::{Array2, Axis};
use numpy::{PyArray2, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyclass]
pub struct SparseMatrix {
    #[pyo3(get)]
    rows: usize,
    #[pyo3(get)]
    cols: usize,
    indptr: Vec<usize>,
    indices: Vec<usize>,
    data: Vec<f32>,
}

#[pymethods]
impl SparseMatrix {
    #[new]
    fn new(
        rows: usize,
        cols: usize,
        indptr: Vec<usize>,
        indices: Vec<usize>,
        data: Vec<f32>,
    ) -> Self {
        SparseMatrix {
            rows,
            cols,
            indptr,
            indices,
            data,
        }
    }

    /// Sparse-Dense Matrix Multiplication: C = A * B
    /// A is (rows x cols) sparse
    /// B is (cols x K) dense
    /// Returns C (rows x K) dense
    #[pyo3(text_signature = "($self, dense)")]
    fn matmul(&self, py: Python, dense: &PyArray2<f32>) -> PyResult<Py<PyArray2<f32>>> {
        let dense_array = unsafe { dense.as_array() };
        let (d_rows, d_cols) = dense_array.dim();

        if self.cols != d_rows {
            return Err(PyValueError::new_err(format!(
                "Dimension mismatch: Sparse({}, {}) vs Dense({}, {})",
                self.rows, self.cols, d_rows, d_cols
            )));
        }

        // Initialize result matrix with zeros
        let mut result = Array2::<f32>::zeros((self.rows, d_cols));

        // Parallel iteration over rows of the sparse matrix
        // result is mutable, split by rows for parallel write access
        result
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut row_out)| {
                let start_idx = self.indptr[i];
                let end_idx = self.indptr[i + 1];

                for k in start_idx..end_idx {
                    let col_idx = self.indices[k];
                    let val = self.data[k];

                    // Let dense_row be the k-th row of B corresponding to column index col_idx of A
                    let dense_row = dense_array.row(col_idx);

                    // Compute row_out += val * dense_row
                    // Manual loop for efficiency and to avoid allocation
                    for j in 0..d_cols {
                        row_out[j] += val * dense_row[j];
                    }
                }
            });

        Ok(PyArray2::from_owned_array(py, result).to_owned())
    }

    /// Create from dense matrix (for testing)
    #[staticmethod]
    fn from_dense(dense: &PyArray2<f32>) -> PyResult<SparseMatrix> {
        let array = unsafe { dense.as_array() };
        let (rows, cols) = array.dim();

        let mut indptr = vec![0];
        let mut indices = Vec::new();
        let mut data = Vec::new();

        for i in 0..rows {
            for j in 0..cols {
                let val = array[[i, j]];
                if val.abs() > 1e-9 {
                    indices.push(j);
                    data.push(val);
                }
            }
            indptr.push(indices.len());
        }

        Ok(SparseMatrix {
            rows,
            cols,
            indptr,
            indices,
            data,
        })
    }
}
