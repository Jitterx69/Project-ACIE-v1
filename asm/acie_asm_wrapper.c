/*
 * C Wrapper for Assembly kernels
 * Provides Python-callable interface via ctypes (simplified version)
 */

#include <stddef.h>
#include <stdint.h>

// Forward declarations of assembly functions
extern void _fast_matrix_multiply_asm(float *A, float *B, float *C, int64_t M,
                                      int64_t N, int64_t K);

extern void _fast_relu_asm(float *input, float *output, int64_t length);
extern void _fast_sigmoid_asm(float *input, float *output, int64_t length);
extern void _minkowski_metric_avx512(float *input, float *output,
                                     int64_t num_points);
extern void _vector_mul_u64_avx512(uint64_t *A, uint64_t *B, uint64_t *C,
                                   int64_t N);
extern void _vector_entropy_term_avx512(float *p, float *out, int64_t N);

// C wrapper functions callable from Python via ctypes
void fast_matmul_wrapper(float *A, float *B, float *C, int64_t M, int64_t N,
                         int64_t K) {
  _fast_matrix_multiply_asm(A, B, C, M, N, K);
}

void fast_relu_wrapper(float *input, float *output, int64_t length) {
  _fast_relu_asm(input, output, length);
}

void fast_sigmoid_wrapper(float *input, float *output, int64_t length) {
  _fast_sigmoid_asm(input, output, length);
}

void fast_minkowski_wrapper(float *input, float *output, int64_t num_points) {
  _minkowski_metric_avx512(input, output, num_points);
}

void vector_mul_u64_wrapper(uint64_t *A, uint64_t *B, uint64_t *C, int64_t N) {
  _vector_mul_u64_avx512(A, B, C, N);
}

void vector_entropy_wrapper(float *p, float *out, int64_t N) {
  _vector_entropy_term_avx512(p, out, N);
}

// Optional: Direct Python extension module (requires Python headers at compile
// time)
#ifdef BUILD_PYTHON_MODULE

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

// Python wrapper for matrix multiply
static PyObject *py_fast_matmul(PyObject *self, PyObject *args) {
  PyArrayObject *A, *B, *C;

  if (!PyArg_ParseTuple(args, "OOO", &A, &B, &C)) {
    return NULL;
  }

  float *A_data = (float *)PyArray_DATA(A);
  float *B_data = (float *)PyArray_DATA(B);
  float *C_data = (float *)PyArray_DATA(C);

  npy_intp *A_dims = PyArray_DIMS(A);
  npy_intp *B_dims = PyArray_DIMS(B);

  _fast_matrix_multiply_asm(A_data, B_data, C_data, (int64_t)A_dims[0],
                            (int64_t)A_dims[1], (int64_t)B_dims[1]);

  Py_RETURN_NONE;
}

// Python wrapper for ReLU
static PyObject *py_fast_relu(PyObject *self, PyObject *args) {
  PyArrayObject *input, *output;

  if (!PyArg_ParseTuple(args, "OO", &input, &output)) {
    return NULL;
  }

  float *in_data = (float *)PyArray_DATA(input);
  float *out_data = (float *)PyArray_DATA(output);
  int64_t length = (int64_t)PyArray_SIZE(input);

  _fast_relu_asm(in_data, out_data, length);

  Py_RETURN_NONE;
}

// Module methods
static PyMethodDef AsmMethods[] = {
    {"fast_matmul", py_fast_matmul, METH_VARARGS, "Fast matrix multiplication"},
    {"fast_relu", py_fast_relu, METH_VARARGS, "Fast ReLU"},
    {NULL, NULL, 0, NULL}};

// Module definition
static struct PyModuleDef asmmodule = {PyModuleDef_HEAD_INIT, "acie_asm",
                                       "ACIE Assembly kernels", -1, AsmMethods};

// Module initialization
PyMODINIT_FUNC PyInit_acie_asm(void) {
  import_array();
  return PyModule_Create(&asmmodule);
}

#endif // BUILD_PYTHON_MODULE
