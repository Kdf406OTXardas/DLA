import numpy as np
from pycuda import driver, compiler, gpuarray, tools
import pycuda.autoinit

kernel_code_template = """ 

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand.h>


__global__ void MatrixMulKernel(float *a, int *b, float *c) 
{   int Pvalue = 0; 
    int num_direction = 3;

    for (int k = 0; k < 25; ++k) {

        int Aelement = a[k]; 
        int Belement = b[k]; 
        int rand_x;

        for (int j = 0; j < 3; ++j){
            Belement = Belement * 1103515245 + 12345;
            if (Belement - (Belement / 3) * 3 == 0)
                rand_x = 1;
            else if (Belement - (Belement / 2) * 2 == 0)
                rand_x = 0;
            else
                rand_x = -1;
        }

        Pvalue = rand_x; 
        c[k] = Pvalue; 
    } 
}"""

MATRIX_SIZE = 5

a_cpu = np.random.randint(2, size=(MATRIX_SIZE,
                                   MATRIX_SIZE)).astype(np.float32)
b_cpu = np.random.randint(1, 19, size=(MATRIX_SIZE,
                                       MATRIX_SIZE)).astype(np.int32)

# a_cpu = np.zeros((MATRIX_SIZE,
#                   MATRIX_SIZE)).astype(np.float32)
# b_cpu = np.zeros((MATRIX_SIZE,
#                   MATRIX_SIZE)).astype(np.float32)


c_cpu = np.dot(a_cpu, b_cpu)
a_gpu = gpuarray.to_gpu(a_cpu)
b_gpu = gpuarray.to_gpu(b_cpu)

c_gpu = gpuarray.empty((MATRIX_SIZE, MATRIX_SIZE), np.float32)

kernel_code = kernel_code_template % {
    'MATRIX_SIZE': MATRIX_SIZE}

mod = compiler.SourceModule(kernel_code)

matrixmul = mod.get_function("MatrixMulKernel")

matrixmul(
    a_gpu, b_gpu,
    c_gpu,
    block=(MATRIX_SIZE, MATRIX_SIZE, 1))

# print("-" * 80)
# print("Matrix A (GPU):")
# print(a_gpu.get())
print("-" * 80)
print("Matrix B (GPU):")
print(b_gpu.get())
print("-" * 80)
print("Matrix C (GPU):")
print(c_gpu.get())

np.allclose(c_cpu, c_gpu.get())

# // next_x = ty
# // next_x = next_x * 1103515245 + 12345
# // x = (next_x % 65536) % 2