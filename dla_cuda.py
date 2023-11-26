import numpy as np
from pycuda import driver, compiler, gpuarray, tools
import pycuda.autoinit

SPACE_VALUE = 3
EDGE_FIELD = 8
END_POROSITY = 10
FOR_NOW_POROSITY = int((EDGE_FIELD - 2) * (EDGE_FIELD - 2) * (EDGE_FIELD - 2))
MAX_COORD = EDGE_FIELD - 2
SIZE_FIELD = EDGE_FIELD ** SPACE_VALUE
middle_index = int(EDGE_FIELD / 2 + EDGE_FIELD * (EDGE_FIELD / 2 + EDGE_FIELD * EDGE_FIELD / 2))
print(f"middle_index {middle_index}")

list_neighbor = [
    # orthogonal
    EDGE_FIELD,
    - EDGE_FIELD,
    1,
    - 1,
    # diagonal
    EDGE_FIELD - 1,
    EDGE_FIELD + 1,
    - EDGE_FIELD - 1,
    - EDGE_FIELD + 1,
    # Y level -1 /(like +0, but with "- EDGE_FIELD * EDGE_FIELD"; condition written for CUDA//orthogonal
    EDGE_FIELD - EDGE_FIELD * EDGE_FIELD,
    - EDGE_FIELD - EDGE_FIELD * EDGE_FIELD,
    1 - EDGE_FIELD * EDGE_FIELD,
    - 1 - EDGE_FIELD * EDGE_FIELD,
    # diagonal
    EDGE_FIELD - 1 - EDGE_FIELD * EDGE_FIELD,
    EDGE_FIELD + 1 - EDGE_FIELD * EDGE_FIELD,
    - EDGE_FIELD - 1 - EDGE_FIELD * EDGE_FIELD,
    - EDGE_FIELD + 1 - EDGE_FIELD * EDGE_FIELD,
    # Central Y level -1
    - EDGE_FIELD * EDGE_FIELD,
    # Y level +1 /(like +0, but with "+ EDGE_FIELD * EDGE_FIELD"; condition written for CUDA//orthogonal
    EDGE_FIELD + EDGE_FIELD * EDGE_FIELD,
    - EDGE_FIELD + EDGE_FIELD * EDGE_FIELD,
    1 + EDGE_FIELD * EDGE_FIELD,
    - 1 + EDGE_FIELD * EDGE_FIELD,
    # diagonal
    EDGE_FIELD - 1 + EDGE_FIELD * EDGE_FIELD,
    EDGE_FIELD + 1 + EDGE_FIELD * EDGE_FIELD,
    - EDGE_FIELD - 1 + EDGE_FIELD * EDGE_FIELD,
    - EDGE_FIELD + 1 + EDGE_FIELD * EDGE_FIELD,
    # Central Y level +1
    EDGE_FIELD * EDGE_FIELD]


kernel_code_template = """ 

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>


__global__ void CalculatingDLA(float *calculating_field, float *array_neighbor)
{
    int k;
    int nbr;
    int value_check_neighbour;
    // Degree of space 3d
    // Input EDGE_FIELD with +2
    int EDGE_FIELD = 8;
    int END_POROSITY = 10; // Input non-porosity [0:100]
    int NOW_POROSITY = 0; // porosity in the calculation process
    int FOR_NOW_POROSITY = (EDGE_FIELD - 2) * (EDGE_FIELD - 2) * (EDGE_FIELD - 2);
    //printf("FOR_NOW_POROSITY %d \\n", FOR_NOW_POROSITY);
    int MAX_COORD = EDGE_FIELD - 2;

// int tid = blockIdx.x*blockDim.x + threadIdx.x;

    int next_x = 1;
    int rand_x;
    //next_x = next_x * 1103515245 + 12345;
    //rand_x = next_x - (next_x / 3) * next_x

    int immobilize_points = 1;
    int counter_immobilize_points = 0;
    int middle_index = EDGE_FIELD / 2 + EDGE_FIELD * (EDGE_FIELD / 2 + EDGE_FIELD * EDGE_FIELD / 2);

    int size_near = 26;
    int counter_in_for = 2;
    int check_counter = 0; // For the limit of calculating new coordinates
    int random_point; // Index of new point coordinates
    int mobile_points = 0;
    int z; // Save z
    int y; // Save y
    int x; // Save x
    
    int counter_while = 0;

//!!! ======= Start main algorithm =======
    while (NOW_POROSITY < END_POROSITY) {
   
        //printf("Calculating of coordinates of a new point\\n");
        
        // Calculating of coordinates of a new point
        if (mobile_points == 0) {
            for (k = 0; k < counter_in_for; k++) {

                // Z-coord new point
                next_x = next_x * 1103515245 + 12345;
                rand_x = next_x - (next_x / 3) * 3;
                if (rand_x < 0){
                    rand_x = - rand_x;
                }
                
                z = middle_index / (EDGE_FIELD * EDGE_FIELD) + (rand_x - 1) * immobilize_points;
                if (z < 1) {
                    z = 1;
                } else if (z > MAX_COORD) {
                    z = MAX_COORD;
                }
        //printf("z %d \\n", z);


                // Y-coord new point
                next_x = next_x * 1103515245 + 12345;
                rand_x = next_x - (next_x / 3) * 3;
                if (rand_x < 0){
                    rand_x = - rand_x;
                }

                y = (middle_index - z * EDGE_FIELD * EDGE_FIELD) / EDGE_FIELD + (rand_x - 1) * immobilize_points;
                if (y < 1) {
                    y = 1;
                } else if (y > MAX_COORD) {
                    y = MAX_COORD;
                }
        //printf("y %d \\n", y);

                // X-coord new point
                next_x = next_x * 1103515245 + 12345;
                rand_x = next_x - (next_x / 3) * 3;
                if (rand_x < 0){
                    rand_x = - rand_x;
                }

                x = middle_index - EDGE_FIELD * (y + EDGE_FIELD * z) + (rand_x - 1) * immobilize_points;
                if (x < 1) {
                    x = 1;
                } else if (x > MAX_COORD) {
                    x = MAX_COORD;
                }
        //printf("x %d \\n", x);

                // Index of new point
                random_point = x + EDGE_FIELD * (y + EDGE_FIELD * z);
                if ((calculating_field[random_point] == 0) || (check_counter == 120)) {
                    break;
                } else {
                    k = 0;
                    check_counter += 1;
                }
            }
            //printf("random_point %d\\n", random_point);
            mobile_points = 1;
            if (calculating_field[random_point] == 1){
                immobilize_points -= 1;
                //printf("check if immobilize - 1 \\n");
            }
            calculating_field[random_point] = 2;
        }
        check_counter = 0;
        counter_immobilize_points = 0;
        

        //printf("Check neighbours\\n");
// ======== Check neighbours ========
        for (nbr = 0; nbr < size_near; nbr++){
            value_check_neighbour = random_point + array_neighbor[nbr];
            if (calculating_field[value_check_neighbour] == 1) {
            calculating_field[random_point] = 1;
            mobile_points = 0;
            counter_immobilize_points += 1;
            }
        }

        // Check porosity
        if (counter_immobilize_points > 0) {
            immobilize_points += 1;
            counter_immobilize_points = 0;
            NOW_POROSITY = immobilize_points * 100 / FOR_NOW_POROSITY;
                        
            printf("NOW_POROSITY %d % \\n", NOW_POROSITY);
            printf("immobilize_points %d \\n", immobilize_points);
        }

        //printf("Moving of points\\n");
// ======== Moving of points ========
        // New Z of moving point
        if (mobile_points > 0) {
            next_x = next_x * 1103515245 + 12345;
            rand_x = next_x - (next_x / 3) * 3;

            z = z + rand_x - 1;
            if (z < 1) {
                z = 1;
            } else if (z > MAX_COORD) {
                z = MAX_COORD;
            }

            // Y of moving point
            next_x = next_x * 1103515245 + 12345;
            rand_x = next_x - (next_x / 3) * 3;
            if (rand_x < 0){
                rand_x = - rand_x;
            }

            y = y + rand_x - 1;
            if (y < 1) {
                y = 1;
            } else if (y > MAX_COORD) {
                y = MAX_COORD;
            }

            // New X of moving point
            next_x = next_x * 1103515245 + 12345;
            rand_x = next_x - (next_x / 3) * 3;
            if (rand_x < 0){
                rand_x = - rand_x;
            }

            x = x + rand_x - 1;
            if (x < 1) {
                x = 1;
            } else if (x > MAX_COORD) {
                x = MAX_COORD;
            }

            // Index of new point
            calculating_field[random_point] = 0;
            random_point = x + EDGE_FIELD * (y + EDGE_FIELD * z);
            calculating_field[random_point] = 2;
        }
        //printf("step_while %d \\n", counter_while);
    }
//!!! ======= End main algorithm =======
}
"""

array_neighbor = np.array(list_neighbor).astype(np.float32)
input_field = np.zeros(SIZE_FIELD).astype(np.float32)
input_field[middle_index] = float(1)

output_field = np.zeros(SIZE_FIELD).astype(np.float32)

print(input_field)
print(middle_index)
print(input_field[middle_index])

field_gpu = gpuarray.to_gpu(input_field)
neighbours = gpuarray.to_gpu(array_neighbor)
# gpu_output = gpuarray.empty((SIZE_FIELD, 1), np.float32)

kernel_code = kernel_code_template
mod = compiler.SourceModule(kernel_code)
DLA_output = mod.get_function("CalculatingDLA")

DLA_output(
    field_gpu,
    neighbours,
    grid=(1, 1, 1),
    block=(1, 1, 1))

# print(f"from cuda {field_gpu}")

counter_render = 0

counter_fin_1 = 0
counter_fin_2 = 0
for i in field_gpu:
    if i == 1:
        counter_fin_1 += 1
    if i == 2:
        counter_fin_2 += 1
#
# arr_3d = field_gpu.reshape((EDGE_FIELD, EDGE_FIELD, EDGE_FIELD)).transpose()
# arr_3d.astype(np.int32)
# print(arr_3d)

# for i in range(SIZE_FIELD):
#     if (counter_render % EDGE_FIELD) == 0:
#         print(f"{counter_render} \n")
#     if counter_render % (EDGE_FIELD * EDGE_FIELD) == 0:
#         print(f"\n Layer_Z: {int((i + 1) / EDGE_FIELD / EDGE_FIELD)} \n")
#     counter_render += 1
#     if field_gpu[i] == 1:
#         print("X", end='    ')
#     elif field_gpu[i] == 2:
#         print("M", end='    ')
#     else:
#         print("o", end='    ')

print(f"\ncounter_1 {counter_fin_1}")
print(f"counter_2 {counter_fin_2}")
print(f"full_size {(EDGE_FIELD - 2) ** SPACE_VALUE}")
print(f"FOR_NOW_POROSITY {FOR_NOW_POROSITY}")

# print(input_field)
# print(len(input_field))
