
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include "cpp/matrix_metal.h"

// matmul wrapper for Python
static PyObject* py_matmul(PyObject* self, PyObject* args) {
	PyObject *a_obj, *b_obj;
	if (!PyArg_ParseTuple(args, "OO", &a_obj, &b_obj))
		return NULL;

	PyArrayObject *a_arr = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
	PyArrayObject *b_arr = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
	if (!a_arr || !b_arr) {
		Py_XDECREF(a_arr);
		Py_XDECREF(b_arr);
		PyErr_SetString(PyExc_ValueError, "Input arrays must be float32");
		return NULL;
	}

	if (PyArray_NDIM(a_arr) != 2 || PyArray_NDIM(b_arr) != 2) {
		Py_DECREF(a_arr);
		Py_DECREF(b_arr);
		PyErr_SetString(PyExc_ValueError, "Input arrays must be 2D");
		return NULL;
	}

	int M = (int)PyArray_DIM(a_arr, 0);
	int K1 = (int)PyArray_DIM(a_arr, 1);
	int K2 = (int)PyArray_DIM(b_arr, 0);
	int N = (int)PyArray_DIM(b_arr, 1);
	if (K1 != K2) {
		Py_DECREF(a_arr);
		Py_DECREF(b_arr);
		PyErr_SetString(PyExc_ValueError, "Inner dimensions must match");
		return NULL;
	}

	npy_intp dims[2] = {M, N};
	PyArrayObject *c_arr = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_FLOAT32);
	if (!c_arr) {
		Py_DECREF(a_arr);
		Py_DECREF(b_arr);
		PyErr_SetString(PyExc_MemoryError, "Could not allocate output array");
		return NULL;
	}

	float* A = (float*)PyArray_DATA(a_arr);
	float* B = (float*)PyArray_DATA(b_arr);
	float* C = (float*)PyArray_DATA(c_arr);
	matmul(A, B, C, M, N, K1);

	Py_DECREF(a_arr);
	Py_DECREF(b_arr);
	return (PyObject*)c_arr;
}

static PyMethodDef MatrixMetalMethods[] = {
	{"matmul", py_matmul, METH_VARARGS, "Matrix multiplication using Metal (or CPU fallback)"},
	{NULL, NULL, 0, NULL}
};

static struct PyModuleDef matrixmetalmodule = {
	PyModuleDef_HEAD_INIT,
	"_matrix_metal",
	"Matrix Metal Extension",
	-1,
	MatrixMetalMethods
};

PyMODINIT_FUNC PyInit__matrix_metal(void) {
	import_array();
	return PyModule_Create(&matrixmetalmodule);
}
