import numpy as np
import pandas as pd
from pycuda import driver, compiler, gpuarray, tools
import pycuda.autoinit
import time
import random as rnd

start_random_position = rnd.randrange(1, 1117, 1)
rand_x = 0

SPACE_VALUE = 3
EDGE_FIELD = 8
END_POROSITY = 5
FOR_NOW_POROSITY = int((EDGE_FIELD - 2) * (EDGE_FIELD - 2) * (EDGE_FIELD - 2))
END_COUNT_POINTS = EDGE_FIELD / 100 * END_POROSITY
print(f"END_COUNT_POINTS {END_COUNT_POINTS}")
MAX_COORD = EDGE_FIELD - 2
SIZE_FIELD = EDGE_FIELD ** SPACE_VALUE
FOR_MIDDLE_INDEX = EDGE_FIELD // 2
MIDDLE_INDEX = int(FOR_MIDDLE_INDEX + EDGE_FIELD * (FOR_MIDDLE_INDEX + EDGE_FIELD * FOR_MIDDLE_INDEX))
print(f"middle_index {MIDDLE_INDEX}")
mobile_points = 0
check_counter = 0
random_point = 0
# properties_field[35] + properties_field[27] * (properties_field[36] +
#                                properties_field[27] * properties_field[37]);
z = int(MIDDLE_INDEX / (EDGE_FIELD * EDGE_FIELD))
y = int((MIDDLE_INDEX / EDGE_FIELD) % EDGE_FIELD)
x = MIDDLE_INDEX % EDGE_FIELD
print(z, y, x)
immobilize_points = 1
counter_immobilize_points = 0

list_neighbor = [
    # orthogonal
    EDGE_FIELD, - EDGE_FIELD, 1, - 1,
    # diagonal
    EDGE_FIELD - 1, EDGE_FIELD + 1, - EDGE_FIELD - 1, - EDGE_FIELD + 1,
    # Y level -1 /(like +0, but with "- EDGE_FIELD * EDGE_FIELD"; condition written for CUDA//orthogonal
    EDGE_FIELD - EDGE_FIELD * EDGE_FIELD, - EDGE_FIELD - EDGE_FIELD * EDGE_FIELD,
    1 - EDGE_FIELD * EDGE_FIELD, - 1 - EDGE_FIELD * EDGE_FIELD,
    # diagonal
    EDGE_FIELD - 1 - EDGE_FIELD * EDGE_FIELD, EDGE_FIELD + 1 - EDGE_FIELD * EDGE_FIELD,
    - EDGE_FIELD - 1 - EDGE_FIELD * EDGE_FIELD, - EDGE_FIELD + 1 - EDGE_FIELD * EDGE_FIELD,
    # Central Y level -1
    - EDGE_FIELD * EDGE_FIELD,
    # Y level +1 /(like +0, but with "+ EDGE_FIELD * EDGE_FIELD"; condition written for CUDA//orthogonal
    EDGE_FIELD + EDGE_FIELD * EDGE_FIELD, - EDGE_FIELD + EDGE_FIELD * EDGE_FIELD,
    1 + EDGE_FIELD * EDGE_FIELD, - 1 + EDGE_FIELD * EDGE_FIELD,
    # diagonal
    EDGE_FIELD - 1 + EDGE_FIELD * EDGE_FIELD, EDGE_FIELD + 1 + EDGE_FIELD * EDGE_FIELD,
    - EDGE_FIELD - 1 + EDGE_FIELD * EDGE_FIELD, - EDGE_FIELD + 1 + EDGE_FIELD * EDGE_FIELD,
    # Central Y level +1
    EDGE_FIELD * EDGE_FIELD]

properties = list_neighbor + [
                              SPACE_VALUE, EDGE_FIELD, END_POROSITY,    # 26, 27, 28
                              FOR_NOW_POROSITY, MAX_COORD, SIZE_FIELD,    # 29, 30, 31
                              MIDDLE_INDEX, check_counter, random_point,    # 32, 33, 34
                              x, y, z,    # 35, 36, 37
                              immobilize_points, counter_immobilize_points,    # 38, 39
                              start_random_position, mobile_points, rand_x     # 40, 41, 42
                              ]
print(properties)


kernel_new_point = """
__global__ void NewPoint(float *calculating_field, float *properties_field)
{
    if (properties_field[41] == 0) {
        for (int k = 0; k < 2; k++) {
    
            // Z-coord new point
            properties_field[40] = __float2int_rn(properties_field[40]) * 1103515245 + 12345;
            properties_field[42] = __float2int_rn(properties_field[40]) -
                                  (__float2int_rn(properties_field[40]) / 3) * 3;
            if (properties_field[42] < 0){
                properties_field[42] = - __float2int_rn(properties_field[42]) - 1;
            } else {
                properties_field[42] = __float2int_rn(properties_field[42]) - 1;
            }
            //printf("rand_x z %d \\n", __float2int_rn(properties_field[42]));
                        
            properties_field[37] = __float2int_rn(properties_field[32]) / __float2int_rn(properties_field[27] *
                                   properties_field[27]) + properties_field[42] * properties_field[38];
            if (properties_field[37] < 1) {
                properties_field[37] = properties_field[30];
            } else if (properties_field[37] > properties_field[30]) {
                properties_field[37] = 1;
            }
            //printf("z %d \\n", __float2int_rn(properties_field[37]));
        
            // Y-coord new point
            properties_field[40] = __float2int_rn(properties_field[40]) * 1103515245 + 12345;
            properties_field[42] = __float2int_rn(properties_field[40]) -
                                  (__float2int_rn(properties_field[40]) / 3) * 3;
            if (properties_field[42] < 0){
                properties_field[42] = - __float2int_rn(properties_field[42]) - 1;
            } else {
                properties_field[42] = __float2int_rn(properties_field[42]) - 1;
            }
            //printf("rand_x y %d \\n", __float2int_rn(properties_field[42]));
        
            properties_field[36] = __float2int_rn(__float2int_rn(properties_field[32] /
                                                  properties_field[27]) / properties_field[27]) +
                                   properties_field[42] * properties_field[38];
            if (properties_field[36] < 1) {
                properties_field[36] = properties_field[30];
            } else if (properties_field[36] > properties_field[30]) {
                properties_field[36] = 1;
            }
            //printf("y %d \\n", __float2int_rn(properties_field[36]));
        
            // X-coord new point
            properties_field[40] = __float2int_rn(properties_field[40]) * 1103515245 + 12345;
            properties_field[42] = __float2int_rn(properties_field[40]) -
                                  (__float2int_rn(properties_field[40]) / 3) * 3;
            if (properties_field[42] < 0){
                properties_field[42] = - __float2int_rn(properties_field[42]) - 1;
            } else {
                properties_field[42] = __float2int_rn(properties_field[42]) - 1;
            }
            //printf("rand_x x %d \\n", __float2int_rn(properties_field[42]));
        
            properties_field[35] = properties_field[32] - __float2int_rn(properties_field[32] /  properties_field[27]) *
                                   properties_field[27] + properties_field[42] * properties_field[38];
            if (properties_field[35] < 1) {
                properties_field[35] = properties_field[30];
            } else if (properties_field[35] > properties_field[30]) {
                properties_field[35] = 1;
            }
            //printf("x %d \\n", __float2int_rn(properties_field[35]));
    
            // Index of new point
            properties_field[34] = properties_field[35] + properties_field[27] * (properties_field[36] +
                                   properties_field[27] * properties_field[37]);
            if ((calculating_field[__float2int_rn(properties_field[34])] == 0) || (properties_field[33] == 120)) {
                if (properties_field[32] > properties_field[31]){
                    properties_field[32] = 0;
                }
                properties_field[32] += 1;
                break;
            } else {
                k = 0;
                properties_field[33] += 1;
            }
        }
        properties_field[41] = 1.0;
        if (calculating_field[__float2int_rn(properties_field[34])] == 1){
            properties_field[38] -= 1;
        }
        calculating_field[__float2int_rn(properties_field[34])] = 2;
        properties_field[33] = 0;
        properties_field[39] = 0;
    }
    __threadfence_system();
}
"""

kernel_check_neighbours = """
__global__ void CheckNeighbour(float *calculating_field, float *properties_field)
{
    int nbr = threadIdx.x;
    if (nbr < 26){
        if (calculating_field[__float2int_rn(properties_field[34] + properties_field[nbr])] == 1) {
        calculating_field[__float2int_rn(properties_field[34])] = 1;
        properties_field[41] = 0;
        properties_field[39] += 1;
        //printf("nbr %d", nbr);
        }
    }
    __threadfence_system();
}
"""

kernel_moving_points = """
__global__ void MovingPoints(float *calculating_field, float *properties_field)
{
    if (properties_field[41] > 0) {
    
        // Z-coord new point
        properties_field[40] = __float2int_rn(properties_field[40]) * 1103515245 + 12345;
        properties_field[42] = __float2int_rn(properties_field[40]) -
                              (__float2int_rn(properties_field[40]) / 3) * 3;
        if (properties_field[42] < 0){
            properties_field[42] = - __float2int_rn(properties_field[42]) - 1;
        } else {
            properties_field[42] = __float2int_rn(properties_field[42]) - 1;
        }
        printf("rand_x z %d \\n", __float2int_rn(properties_field[42]));
                        
        properties_field[37] = __float2int_rn(properties_field[32]) / __float2int_rn(properties_field[27] *
                               properties_field[27]) + properties_field[42];
        if (properties_field[37] < 1) {
            properties_field[37] = properties_field[30];
        } else if (properties_field[37] > properties_field[30]) {
            properties_field[37] = 1;
        }
        printf("z %d \\n", __float2int_rn(properties_field[37]));
        
        // Y-coord new point
        properties_field[40] = __float2int_rn(properties_field[40]) * 1103515245 + 12345;
        properties_field[42] = __float2int_rn(properties_field[40]) -
                                (__float2int_rn(properties_field[40]) / 3) * 3;
        if (properties_field[42] < 0){
            properties_field[42] = - __float2int_rn(properties_field[42]) - 1;
        } else {
            properties_field[42] = __float2int_rn(properties_field[42]) - 1;
        }
        printf("rand_x y %d \\n", __float2int_rn(properties_field[42]));
        
        properties_field[36] = __float2int_rn(__float2int_rn(properties_field[32] /
                                              properties_field[27]) / properties_field[27]) + properties_field[42];
        if (properties_field[36] < 1) {
            properties_field[36] = properties_field[30];
        } else if (properties_field[36] > properties_field[30]) {
            properties_field[36] = 1;
        }
        printf("y %d \\n", __float2int_rn(properties_field[36]));
        
        // X-coord new point
        properties_field[40] = __float2int_rn(properties_field[40]) * 1103515245 + 12345;
        properties_field[42] = __float2int_rn(properties_field[40]) -
                              (__float2int_rn(properties_field[40]) / 3) * 3;
        if (properties_field[42] < 0){
            properties_field[42] = - __float2int_rn(properties_field[42]) - 1;
        } else {
            properties_field[42] = __float2int_rn(properties_field[42]) - 1;
        }
        printf("rand_x x %d \\n", __float2int_rn(properties_field[42]));
        
        properties_field[35] = properties_field[32] - __float2int_rn(properties_field[32] /  properties_field[27]) *
                               properties_field[27] + properties_field[42];
        if (properties_field[35] < 1) {
            properties_field[35] = properties_field[30];
        } else if (properties_field[35] > properties_field[30]) {
            properties_field[35] = 1;
        }
        printf("x %d \\n", __float2int_rn(properties_field[35]));
        calculating_field[__float2int_rn(properties_field[34])] = 0;
        properties_field[34] = properties_field[35] + properties_field[27] * (properties_field[36] +
                               properties_field[27] * properties_field[37]);
        calculating_field[__float2int_rn(properties_field[34])] = 2;
    }
    __threadfence_system();
}
"""


input_field = np.zeros(SIZE_FIELD).astype(np.float32)
input_field[MIDDLE_INDEX] = float(1)
# print(input_field)
input_properties = np.array(properties).astype(np.float32)

field_gpu = gpuarray.to_gpu(input_field)
properties_gpu = gpuarray.to_gpu(input_properties)

# print(properties_gpu)
mod_1 = compiler.SourceModule(kernel_new_point)
DLA_new_point = mod_1.get_function("NewPoint")

mod_2 = compiler.SourceModule(kernel_check_neighbours)
DLA_check_neighbours = mod_2.get_function("CheckNeighbour")

mod_3 = compiler.SourceModule(kernel_moving_points)
DLA_moving_points = mod_3.get_function("MovingPoints")

st_time = time.time()

for i in range(2):
    print('initialization')
    DLA_new_point(
        field_gpu,
        properties_gpu,
        grid=(1, 1, 1),
        block=(1, 1, 1))
    # print(properties_gpu)
    # arr_3d = field_gpu.reshape((EDGE_FIELD, EDGE_FIELD, EDGE_FIELD)).transpose().astype(np.int32)
    # print(arr_3d)

    print('check')
    DLA_check_neighbours(
        field_gpu,
        properties_gpu,
        grid=(1, 1, 1),
        block=(26, 1, 1))


# while field_gpu[38] < END_COUNT_POINTS:
#     print('initialization')
#     DLA_new_point(
#         field_gpu,
#         properties_gpu,
#         grid=(1, 1, 1),
#         block=(1, 1, 1))
#     # print(properties_gpu)
#     # arr_3d = field_gpu.reshape((EDGE_FIELD, EDGE_FIELD, EDGE_FIELD)).transpose().astype(np.int32)
#     # print(arr_3d)
#
#     print('check')
#     DLA_check_neighbours(
#         field_gpu,
#         properties_gpu,
#         grid=(1, 1, 1),
#         block=(26, 1, 1))
#     # print(field_gpu)
#     # arr_3d = field_gpu.reshape((EDGE_FIELD, EDGE_FIELD, EDGE_FIELD)).transpose().astype(np.int32)
#     # print(arr_3d)
#
#     print('moving')
#     DLA_moving_points(
#         field_gpu,
#         properties_gpu,
#         grid=(1, 1, 1),
#         block=(1, 1, 1))
#     # print(field_gpu)
#     # arr_3d = field_gpu.reshape((EDGE_FIELD, EDGE_FIELD, EDGE_FIELD)).transpose().astype(np.int32)
#     # print(arr_3d)


arr_3d = field_gpu.reshape((EDGE_FIELD, EDGE_FIELD, EDGE_FIELD)).transpose().astype(np.int32)
print(arr_3d)
