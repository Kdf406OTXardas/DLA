import numpy as np
import pandas as pd
from pycuda import driver, compiler, gpuarray, tools
import pycuda.autoinit
import time

SPACE_VALUE = 3
EDGE_FIELD = 20
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
#include <time.h>



__global__ void CalculatingDLA(float *calculating_field)
{
//    int i = threadIdx.x;
    int k;
    int nbr = threadIdx.x;
    int value_check_neighbour;
    // Degree of space 3d
    // Input EDGE_FIELD with +2
    int EDGE_FIELD = 20;
    int END_POROSITY = 10; // Input non-porosity [0:100]
    int NOW_POROSITY = 0; // porosity in the calculation process
    int FOR_NOW_POROSITY = (EDGE_FIELD - 2) * (EDGE_FIELD - 2) * (EDGE_FIELD - 2);
    //printf("FOR_NOW_POROSITY %d \\n", FOR_NOW_POROSITY);
    int MAX_COORD = EDGE_FIELD - 2;
    int SIZE_FIELD = EDGE_FIELD * EDGE_FIELD * EDGE_FIELD;

// int tid = blockIdx.x*blockDim.x + threadIdx.x;

    int next_x = clock() / 1000;
    int rand_x;
    //next_x = next_x * 1103515245 + 12345;
    //rand_x = next_x - (next_x / 3) * next_x

    int immobilize_points = 1;
    int counter_immobilize_points = 0;
    int middle_index = EDGE_FIELD / 2 + EDGE_FIELD * (EDGE_FIELD / 2 + EDGE_FIELD * EDGE_FIELD / 2);


    int size_near = 26;
    int array_neighbor [26];
    // Y level +0; condition written for CUDA
    //orthogonal
    array_neighbor[0] = EDGE_FIELD;
    array_neighbor[1] = - EDGE_FIELD;
    array_neighbor[2] = 1;
    array_neighbor[3] = - 1;
    //diagonal
    array_neighbor[4] = EDGE_FIELD - 1;
    array_neighbor[5] = EDGE_FIELD + 1;
    array_neighbor[6] = - EDGE_FIELD - 1;
    array_neighbor[7] = - EDGE_FIELD + 1;
    // Y level -1 /(like +0, but with "- EDGE_FIELD * EDGE_FIELD"; condition written for CUDA//orthogonal
    array_neighbor[8] = EDGE_FIELD - EDGE_FIELD * EDGE_FIELD;
    array_neighbor[9] = - EDGE_FIELD - EDGE_FIELD * EDGE_FIELD;
    array_neighbor[10] = 1 - EDGE_FIELD * EDGE_FIELD;
    array_neighbor[11] = - 1 - EDGE_FIELD * EDGE_FIELD;
    //diagonal
    array_neighbor[12] = EDGE_FIELD - 1 - EDGE_FIELD * EDGE_FIELD;
    array_neighbor[13] = EDGE_FIELD + 1 - EDGE_FIELD * EDGE_FIELD;
    array_neighbor[14] = - EDGE_FIELD - 1 - EDGE_FIELD * EDGE_FIELD;
    array_neighbor[15] = - EDGE_FIELD + 1 - EDGE_FIELD * EDGE_FIELD;
    // Central Y level -1
    array_neighbor[16] = - EDGE_FIELD * EDGE_FIELD;
    // Y level +1 /(like +0, but with "+ EDGE_FIELD * EDGE_FIELD"; condition written for CUDA//orthogonal
    array_neighbor[17] = EDGE_FIELD + EDGE_FIELD * EDGE_FIELD;
    array_neighbor[18] = - EDGE_FIELD + EDGE_FIELD * EDGE_FIELD;
    array_neighbor[19] = 1 + EDGE_FIELD * EDGE_FIELD;
    array_neighbor[20] = - 1 + EDGE_FIELD * EDGE_FIELD;
    //diagonal
    array_neighbor[21] = EDGE_FIELD - 1 + EDGE_FIELD * EDGE_FIELD;
    array_neighbor[22] = EDGE_FIELD + 1 + EDGE_FIELD * EDGE_FIELD;
    array_neighbor[23] = - EDGE_FIELD - 1 + EDGE_FIELD * EDGE_FIELD;
    array_neighbor[24] = - EDGE_FIELD + 1 + EDGE_FIELD * EDGE_FIELD;
    // Central Y level +1
    array_neighbor[25] = EDGE_FIELD * EDGE_FIELD;
    
    int counter_in_for = 2;
    int check_counter = 0; // For the limit of calculating new coordinates
    int random_point; // Index of new point coordinates
    int mobile_points = 0;
    int z; // Save z
    int y; // Save y
    int x; // Save x
    

//    unsigned int start_time;
//    unsigned int end_time;
//    start_time = clock() / CLOCKS_PER_SEC;
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
                    z = MAX_COORD;
                } else if (z > MAX_COORD) {
                    z = 1;
                }
        //printf("z %d \\n", z);


                // Y-coord new point
                next_x = next_x * 1103515245 + 12345;
                rand_x = next_x - (next_x / 3) * 3;
                if (rand_x < 0){
                    rand_x = - rand_x;
                }

                y = middle_index / EDGE_FIELD / EDGE_FIELD + (rand_x - 1) * immobilize_points;
                if (y < 1) {
                    y = MAX_COORD;
                } else if (y > MAX_COORD) {
                    y = 1;
                }
        //printf("y %d \\n", y);

                // X-coord new point
                next_x = next_x * 1103515245 + 12345;
                rand_x = next_x - (next_x / 3) * 3;
                if (rand_x < 0){
                    rand_x = - rand_x;
                }

                x = middle_index - middle_index / EDGE_FIELD * EDGE_FIELD + (rand_x - 1) * immobilize_points;
                if (x < 1) {
                    x = MAX_COORD;
                } else if (x > MAX_COORD) {
                    x = 1;
                }
        //printf("x %d \\n", x);

                // Index of new point
                random_point = x + EDGE_FIELD * (y + EDGE_FIELD * z);
                if ((calculating_field[random_point] == 0) || (check_counter == 120)) {
                    if (middle_index > SIZE_FIELD){
                    middle_index = 0;
                    }
                    middle_index += 1;
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
// -------- Append new flows START --------
        //if (nbr < 27){
        //    value_check_neighbour = random_point + array_neighbor[nbr - 1];
        //    if (calculating_field[value_check_neighbour] == 1) {
        //    calculating_field[random_point] = 1;
        //    mobile_points = 0;
        //    counter_immobilize_points += 1;
        //    }
        //}
        //__syncthreads();
        for (nbr = 0; nbr < size_near; nbr++){
            value_check_neighbour = random_point + array_neighbor[nbr];
            if (calculating_field[value_check_neighbour] == 1) {
            calculating_field[random_point] = 1;
            mobile_points = 0;
            counter_immobilize_points += 1;
           }
        }
// -------- Append new flows END --------

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
            if (rand_x < 0){
                rand_x = - rand_x;
            }

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
        printf("x %d ", x);
        printf("y %d ", y);
        printf("z %d \\n", z);
        printf("immobilize_points %d \\n", immobilize_points);
        //printf("step_while %d \\n", counter_while);
    }
//    end_time = clock() / CLOCKS_PER_SEC;
//!!! ======= End main algorithm =======
//    printf("end_time - start_time %d\\n", (end_time - start_time));
}
"""

# array_neighbor = np.array(list_neighbor).astype(np.float32)
input_field = np.zeros(SIZE_FIELD).astype(np.float32)
input_field[middle_index] = float(1)

output_field = np.zeros(SIZE_FIELD).astype(np.float32)

print(input_field)
print(middle_index)
print(input_field[middle_index])

field_gpu = gpuarray.to_gpu(input_field)
# neighbours = gpuarray.to_gpu(array_neighbor)
# gpu_output = gpuarray.empty((SIZE_FIELD, 1), np.float32)

kernel_code = kernel_code_template
mod = compiler.SourceModule(kernel_code)
DLA_output = mod.get_function("CalculatingDLA")

st_time = time.time()

DLA_output(
    field_gpu,
    # neighbours,
    grid=(1, 1, 1),
    block=(1, 1, 1))

en_time = time.time()
print(f"en_time - st_time {en_time - st_time}")
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

df_export = np.array(field_gpu)

df = pd.DataFrame(df_export, columns=['property'])
df['z'] = (df.index / (EDGE_FIELD * EDGE_FIELD)).astype(int)
df['y'] = (df.index / EDGE_FIELD).astype(int) - ((df.index / EDGE_FIELD / EDGE_FIELD) * EDGE_FIELD).astype(int)
df['x'] = (df.index - EDGE_FIELD * (df.index / EDGE_FIELD).astype(int)).astype(int)
print(df)
df.to_csv(f"/home/natkachov/datasets/export_from_dla_cuda/dla_cuda_{EDGE_FIELD}.csv", index=False)
