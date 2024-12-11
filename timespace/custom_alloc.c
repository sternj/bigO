#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#ifdef __APPLE__
#include <malloc/malloc.h>  // For malloc_size on macOS
#define malloc_usable_size malloc_size
#else
#include <malloc.h>         // For malloc_usable_size on Linux
#endif

// Global variable for the dilation factor
static double DILATION_FACTOR = 1.0;  // Default value

#include <stdint.h>

// Seed for the RNG (must be non-zero)
static uint32_t rng_state = 1;  // Default seed

// Fast Xorshift RNG function
uint32_t xorshift32() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

// Function to seed the RNG
void seed_xorshift(uint32_t seed) {
    if (seed != 0) {
        rng_state = seed;
    } else {
        rng_state = 1;  // Avoid zero seed
    }
}


// Function declarations
void custom_free(void *ctx, void *ptr);
void *custom_malloc(void *ctx, size_t size);
void *custom_calloc(void *ctx, size_t nelems, size_t size);
void *custom_realloc(void *ctx, void *ptr, size_t size);

// Set the dilation factor
static PyObject *set_dilation_factor(PyObject *self, PyObject *args) {
    double factor;
    if (!PyArg_ParseTuple(args, "d", &factor)) {
      printf("Warning: couldn't parse dilation factor.\n");
      DILATION_FACTOR = 1.0;
      return NULL;  // Parse error
    }
    DILATION_FACTOR = factor;
    Py_RETURN_NONE;
}

// Perform probabilistic dilation
size_t apply_dilation(size_t size, double dilation_factor) {
    double ceil_factor = ceil(dilation_factor);
    double prob = dilation_factor / ceil_factor;
    double r_unif = (double)xorshift32() / (double)UINT32_MAX;
    //    printf("[DEBUG] Applying dilation: prob=%f, random=%f\n", prob, r_unif);
    if (r_unif < prob) {
        return (size_t)(ceil_factor * size);
    }
    return size;
}

static PyMemAllocatorEx origAlloc;

static PyMemAllocatorEx alloc = {
  .ctx = NULL,
  .malloc = custom_malloc,
  .calloc = custom_calloc,
  .realloc = custom_realloc,
  .free = custom_free,
};

// Custom malloc implementation
void *custom_malloc(void *ctx, size_t size) {
    size_t dilated_size = apply_dilation(size, DILATION_FACTOR);
    //  printf("[DEBUG] Allocating %zu bytes (dilated to %zu bytes with factor %.2f).\n",
    //       size, dilated_size, DILATION_FACTOR);
    return origAlloc.malloc(ctx, dilated_size);
}

// Custom calloc implementation
void *custom_calloc(void *ctx, size_t nelem, size_t elsize) {
    size_t size = nelem * elsize;
    size_t dilated_size = apply_dilation(size, DILATION_FACTOR);
    //printf("[DEBUG] Allocating %zu bytes (dilated to %zu bytes with factor %.2f).\n",
    //       size, dilated_size, dilation_factor);
    return origAlloc.calloc(ctx, dilated_size, 1);
}

// Custom realloc implementation
void *custom_realloc(void *ctx, void *ptr, size_t size) {
    if (ptr == NULL) {
        return custom_malloc(ctx, size);
    }
    if (size == 0) {
        custom_free(ctx, ptr);
        return NULL;
    }
    return origAlloc.realloc(ctx, ptr, size);
}

// Custom free implementation
void custom_free(void *ctx, void *ptr) {
  origAlloc.free(ctx, ptr);
}

// Set the custom allocator
void set_custom_allocator() {
    PyMem_GetAllocator(PYMEM_DOMAIN_OBJ, &origAlloc);
    PyMem_SetAllocator(PYMEM_DOMAIN_MEM, &alloc);
    PyMem_SetAllocator(PYMEM_DOMAIN_OBJ, &alloc);
}

// Python methods
static PyMethodDef CustomAllocMethods[] = {
    {"set_dilation_factor", set_dilation_factor, METH_VARARGS, "Set the dilation factor."},
    {NULL, NULL, 0, NULL}  // Sentinel
};

// Module definition
static struct PyModuleDef customalloc_module = {
    PyModuleDef_HEAD_INIT,
    "customalloc",
    "Custom memory allocator with dilation",
    -1,
    CustomAllocMethods
};

// Module initialization function
PyMODINIT_FUNC PyInit_customalloc(void) {
    seed_xorshift(time(NULL));  // Seed the random number generator
    set_custom_allocator();
    return PyModule_Create(&customalloc_module);
}
