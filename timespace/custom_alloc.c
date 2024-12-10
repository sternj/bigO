#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#ifdef __APPLE__
#include <malloc/malloc.h>  // For malloc_size on macOS
#define malloc_usable_size malloc_size
#else
#include <malloc.h>         // For malloc_usable_size on Linux
#endif

// Function declarations
void custom_free(void *ctx, void *ptr);

// Helper function to get the dilation factor from a global Python variable
#include <Python.h>
#include <stdio.h>

// Helper function to safely retrieve the dilation factor
#include <Python.h>
#include <stdio.h>

// Helper function to safely retrieve the dilation factor
double get_dilation_factor() {
    PyMemAllocatorEx prev_alloc;
    PyMem_GetAllocator(PYMEM_DOMAIN_RAW, &prev_alloc);  // Save current allocator
    PyMemAllocatorEx null_alloc = {NULL, NULL, NULL, NULL, NULL};
    PyMem_SetAllocator(PYMEM_DOMAIN_RAW, &null_alloc);  // Disable custom allocator
  
    PyGILState_STATE gstate = PyGILState_Ensure();  // Acquire the GIL
    double dilation_factor = 1.0;  // Default value

    // Import the timespace module
    PyObject *module = PyImport_ImportModule("timespace");
    if (module) {
        // Retrieve the DILATION_FACTOR attribute
        PyObject *dilation_factor_obj = PyObject_GetAttrString(module, "DILATION_FACTOR");
        if (dilation_factor_obj && PyFloat_Check(dilation_factor_obj)) {
            dilation_factor = PyFloat_AsDouble(dilation_factor_obj);
            Py_DECREF(dilation_factor_obj);  // Decrement reference count
        } else {
            fprintf(stderr, "[ERROR] DILATION_FACTOR not found or invalid in 'timespace'.\n");
        }
        Py_DECREF(module);  // Decrement module reference count
    } else {
        PyErr_Print();  // Print the Python error if the module import fails
        fprintf(stderr, "[ERROR] Failed to import 'timespace' module.\n");
    }
    printf("[DEBUG] Dilation factor = %f\n", dilation_factor);
    
    PyGILState_Release(gstate);  // Release the GIL
    // Restore the custom allocator
    PyMem_SetAllocator(PYMEM_DOMAIN_RAW, &prev_alloc);
    return dilation_factor;
}


// Perform probabilistic dilation
size_t apply_dilation(size_t size, double dilation_factor) {
    double ceil_factor = ceil(dilation_factor);
    double prob = dilation_factor / ceil_factor;
    double r_unif = (double) rand() / RAND_MAX;
    printf("probably? %f (%f)\n", prob, r_unif);
    if (r_unif < prob) {
        return (size_t)(ceil_factor * size);
    }
    return size;
}

// Reverse the dilation to approximate the original size
size_t reverse_dilation(size_t size, double dilation_factor) {
    double ceil_factor = ceil(dilation_factor);
    double prob = dilation_factor / ceil_factor;

    // Approximate the original size based on expected dilation
    return (size_t)(size / ((prob * ceil_factor) + ((1.0 - prob) * 1.0)));
}

// Custom malloc implementation
void *custom_malloc(void *ctx, size_t size) {
    double dilation_factor = get_dilation_factor();
    size_t dilated_size = apply_dilation(size, dilation_factor);

    printf("[DEBUG] Allocating %zu bytes (dilated to %zu bytes with factor %.2f).\n",
           size, dilated_size, dilation_factor);

    return malloc(dilated_size);
}

// Custom calloc implementation
void *custom_calloc(void *ctx, size_t nelem, size_t elsize) {
    size_t size = nelem * elsize;
    double dilation_factor = get_dilation_factor();
    size_t dilated_size = apply_dilation(size, dilation_factor);

    printf("[DEBUG] Allocating %zu bytes (dilated to %zu bytes with factor %.2f).\n",
           size, dilated_size, dilation_factor);

    return calloc(dilated_size, 1);
}

// Custom realloc implementation
void *custom_realloc(void *ctx, void *ptr, size_t size) {
    if (ptr == NULL) {
        // If no existing pointer, this behaves like malloc
        return custom_malloc(ctx, size);
    }
    if (size == 0) {
        // If size is 0, this behaves like free
        custom_free(ctx, ptr);
        return NULL;
    }

    double dilation_factor = get_dilation_factor();

    // Reverse the previous dilation to approximate the original allocation size
    size_t current_size = malloc_usable_size(ptr);  // Current usable size of the allocated block
    size_t original_size = reverse_dilation(current_size, dilation_factor);

    printf("[DEBUG] Original size estimated as %zu bytes from allocated %zu bytes.\n",
           original_size, current_size);

    // Apply dilation for the new requested size
    size_t dilated_size = apply_dilation(size, dilation_factor);

    printf("[DEBUG] Reallocating from %zu bytes to %zu bytes (dilated to %zu bytes with factor %.2f).\n",
           original_size, size, dilated_size, dilation_factor);

    return realloc(ptr, dilated_size);
}

// Custom free implementation
void custom_free(void *ctx, void *ptr) {
    printf("[DEBUG] Freeing memory at %p.\n", ptr);
    free(ptr);
}

// Set the custom allocator
void set_custom_allocator() {
    PyMemAllocatorEx alloc = {
        .ctx = NULL,
        .malloc = custom_malloc,
        .calloc = custom_calloc,
        .realloc = custom_realloc,
        .free = custom_free,
    };
    PyMem_SetAllocator(PYMEM_DOMAIN_RAW, &alloc);
}

// Module initialization function
static PyModuleDef customalloc_module = {
    PyModuleDef_HEAD_INIT,
    "customalloc",
    "Custom memory allocator with dilation",
    -1,
    NULL, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit_customalloc(void) {
    // Make a different seed for every execution
    srand(time(NULL));
    set_custom_allocator();
    return PyModule_Create(&customalloc_module);
}
