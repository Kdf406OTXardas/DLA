import numpy as np
from pycuda import driver, compiler, gpuarray, tools
import pycuda.autoinit

SPACE_VALUE = 3
EDGE_FIELD = 10
END_POROSITY = 10
FOR_NOW_POROSITY = int((EDGE_FIELD - 2) * (EDGE_FIELD - 2) * (EDGE_FIELD - 2))
MAX_COORD = EDGE_FIELD - 2
SIZE_FIELD = EDGE_FIELD ** SPACE_VALUE
middle_index = int(EDGE_FIELD / 2 + EDGE_FIELD * (EDGE_FIELD / 2 + EDGE_FIELD * EDGE_FIELD / 2))
print(f"middle_index {middle_index}")
kernel_code_template = """ 

#include <stdio.h>
#include <stdlib.h>


__global__ void CalculatingDLA(float *calculating_field)
{
    int k;
    // Degree of space 3d
    // Input EDGE_FIELD with +2
    int EDGE_FIELD = 10;
    int END_POROSITY = 20; // Input non-porosity [0:100]
    int NOW_POROSITY = 0; // porosity in the calculation process
    int FOR_NOW_POROSITY = (EDGE_FIELD - 2) * (EDGE_FIELD - 2) * (EDGE_FIELD - 2);
    int MAX_COORD = EDGE_FIELD - 2;

    int next_x = 11;
    int rand_x;
    //next_x = next_x * 1103515245 + 12345;
    //rand_x = next_x - (next_x / 3) * next_x

    int immobilize_points = 1;
    int counter_immobilize_points = 0;
    int middle_index = EDGE_FIELD / 2 + EDGE_FIELD * (EDGE_FIELD / 2 + EDGE_FIELD * EDGE_FIELD / 2);


    int counter_in_for = 2;
    int check_counter = 0; // For the limit of calculating new coordinates
    int random_point; // Index of new point coordinates
    int mobile_points = 0;
    int z; // Save z
    int y; // Save y
    int x; // Save x


//!!! ======= Start main algorithm =======
    while (NOW_POROSITY < END_POROSITY) {

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

                // Index of new point
                random_point = x + EDGE_FIELD * (y + EDGE_FIELD * z);

                    if ((calculating_field[random_point] == 0) || (check_counter == 120)) {
                        break;
                    } else {
                        k = 0;
                        check_counter += 1;
                    }
                    mobile_points = 1;
            }
        }
        check_counter = 0;

        // Check porosity
        if (counter_immobilize_points > 0) {
            immobilize_points += 1;
            counter_immobilize_points = 0;
            NOW_POROSITY = immobilize_points * 100 / FOR_NOW_POROSITY;
        }

        if (random_point == middle_index){
        immobilize_points -= 1;
        }

        counter_immobilize_points = 0;
        calculating_field[random_point] = 2;

// ======== Check neighbours ========
        // Y level +0; condition written for CUDA
        //orthogonal
        if (calculating_field[random_point + EDGE_FIELD] == 1) {
            calculating_field[random_point] = 1;
            mobile_points = 0;
            counter_immobilize_points += 1;
        }
        if (calculating_field[random_point - EDGE_FIELD] == 1) {
            calculating_field[random_point] = 1;
            mobile_points = 0;
            counter_immobilize_points += 1;
        }
        if (calculating_field[random_point + 1] == 1) {
            calculating_field[random_point] = 1;
            mobile_points = 0;
            counter_immobilize_points += 1;
        }
        if (calculating_field[random_point - 1] == 1) {
            calculating_field[random_point] = 1;
            mobile_points = 0;
            counter_immobilize_points += 1;
        }
        //diagonal
        if (calculating_field[random_point + EDGE_FIELD - 1] == 1) {
            calculating_field[random_point] = 1;
            mobile_points = 0;
            counter_immobilize_points += 1;
        }
        if (calculating_field[random_point + EDGE_FIELD + 1] == 1) {
            calculating_field[random_point] = 1;
            mobile_points = 0;
            counter_immobilize_points += 1;
        }
        if (calculating_field[random_point - EDGE_FIELD - 1] == 1) {
            calculating_field[random_point] = 1;
            mobile_points = 0;
            counter_immobilize_points += 1;
        }
        if (calculating_field[random_point - EDGE_FIELD + 1] == 1) {
            calculating_field[random_point] = 1;
            mobile_points = 0;
            counter_immobilize_points += 1;
        }

        // Y level -1 /(like +0, but with - EDGE_FIELD * EDGE_FIELD; condition written for CUDA
        //orthogonal
        if (calculating_field[random_point + EDGE_FIELD - EDGE_FIELD * EDGE_FIELD] == 1) {
            calculating_field[random_point] = 1;
            mobile_points = 0;
            counter_immobilize_points += 1;
        }
        if (calculating_field[random_point - EDGE_FIELD - EDGE_FIELD * EDGE_FIELD] == 1) {
            calculating_field[random_point] = 1;
            mobile_points = 0;
            counter_immobilize_points += 1;
        }
        if (calculating_field[random_point + 1 - EDGE_FIELD * EDGE_FIELD] == 1) {
            calculating_field[random_point] = 1;
            mobile_points = 0;
            counter_immobilize_points += 1;
        }
        if (calculating_field[random_point - 1 - EDGE_FIELD * EDGE_FIELD] == 1) {
            calculating_field[random_point] = 1;
            mobile_points = 0;
            counter_immobilize_points += 1;
        }
        //diagonal
        if (calculating_field[random_point + EDGE_FIELD - 1 - EDGE_FIELD * EDGE_FIELD] == 1) {
            calculating_field[random_point] = 1;
            mobile_points = 0;
            counter_immobilize_points += 1;
        }
        if (calculating_field[random_point + EDGE_FIELD + 1 - EDGE_FIELD * EDGE_FIELD] == 1) {
            calculating_field[random_point] = 1;
            mobile_points = 0;
            counter_immobilize_points += 1;
        }
        if (calculating_field[random_point - EDGE_FIELD - 1 - EDGE_FIELD * EDGE_FIELD] == 1) {
            calculating_field[random_point] = 1;
            mobile_points = 0;
            counter_immobilize_points += 1;
        }
        if (calculating_field[random_point - EDGE_FIELD + 1 - EDGE_FIELD * EDGE_FIELD] == 1) {
            calculating_field[random_point] = 1;
            mobile_points = 0;
            counter_immobilize_points += 1;
        }
        // Central Y level -1
        if (calculating_field[random_point - EDGE_FIELD * EDGE_FIELD] == 1) {
            calculating_field[random_point] = 1;
            mobile_points = 0;
            counter_immobilize_points += 1;
        }

        // Y level +1 /(like +0, but with + EDGE_FIELD * EDGE_FIELD; condition written for CUDA
        //orthogonal
        if (calculating_field[random_point + EDGE_FIELD + EDGE_FIELD * EDGE_FIELD] == 1) {
            calculating_field[random_point] = 1;
            mobile_points = 0;
            counter_immobilize_points += 1;
        }
        if (calculating_field[random_point - EDGE_FIELD + EDGE_FIELD * EDGE_FIELD] == 1) {
            calculating_field[random_point] = 1;
            mobile_points = 0;
            counter_immobilize_points += 1;
        }
        if (calculating_field[random_point + 1 + EDGE_FIELD * EDGE_FIELD] == 1) {
            calculating_field[random_point] = 1;
            mobile_points = 0;
            counter_immobilize_points += 1;
        }
        if (calculating_field[random_point - 1 + EDGE_FIELD * EDGE_FIELD] == 1) {
            calculating_field[random_point] = 1;
            mobile_points = 0;
            counter_immobilize_points += 1;
        }
        //diagonal
        if (calculating_field[random_point + EDGE_FIELD - 1 + EDGE_FIELD * EDGE_FIELD] == 1) {
            calculating_field[random_point] = 1;
            mobile_points = 0;
            counter_immobilize_points += 1;
        }
        if (calculating_field[random_point + EDGE_FIELD + 1 + EDGE_FIELD * EDGE_FIELD] == 1) {
            calculating_field[random_point] = 1;
            mobile_points = 0;
            counter_immobilize_points += 1;
        }
        if (calculating_field[random_point - EDGE_FIELD - 1 + EDGE_FIELD * EDGE_FIELD] == 1) {
            calculating_field[random_point] = 1;
            mobile_points = 0;
            counter_immobilize_points += 1;
        }
        if (calculating_field[random_point - EDGE_FIELD + 1 + EDGE_FIELD * EDGE_FIELD] == 1) {
            calculating_field[random_point] = 1;
            mobile_points = 0;
            counter_immobilize_points += 1;
        }
        // Central Y level +1
        if (calculating_field[random_point + EDGE_FIELD * EDGE_FIELD] == 1) {
            calculating_field[random_point] = 1;
            mobile_points = 0;
            counter_immobilize_points += 1;
        }

        // Check porosity
        if (counter_immobilize_points > 0) {
            immobilize_points += 1;
            counter_immobilize_points = 0;
            NOW_POROSITY = immobilize_points * 100 / FOR_NOW_POROSITY;
        }

// ======== Moving of points ========
        // New Z of moving point
        if (mobile_points > 0) {
            // X of moving point
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
    }
//!!! ======= End main algorithm =======
}
"""

input_field = np.zeros(SIZE_FIELD).astype(np.float32)
input_field[middle_index] = float(1)

output_field = np.zeros(SIZE_FIELD).astype(np.float32)

print(input_field)
print(middle_index)
print(input_field[middle_index])

field_gpu = gpuarray.to_gpu(input_field)
gpu_output = gpuarray.empty((SIZE_FIELD, 1), np.float32)

kernel_code = kernel_code_template

mod = compiler.SourceModule(kernel_code)

DLA_output = mod.get_function("CalculatingDLA")

DLA_output(
    field_gpu,
    block=(10, 10, 1))

print(f"from cuda {field_gpu}")

counter_render = 0

arr_3d = field_gpu.reshape((EDGE_FIELD, EDGE_FIELD, EDGE_FIELD)).transpose()
arr_3d.astype(np.int32)
print(arr_3d)

# for i in range(SIZE_FIELD):
#     if (counter_render % EDGE_FIELD) == 0:
#         print(f"{counter_render} \n")
#     if counter_render % (EDGE_FIELD * EDGE_FIELD) == 0:
#         print(f"\n Layer_Z: {int((i + 1) / EDGE_FIELD / EDGE_FIELD)} \n")
#     counter_render += 1
#     if field_gpu[i] == 1:
#         print("X")
#     elif field_gpu[i] == 2:
#         print("M")
#     else:
#         print("o")
# print(input_field)
# print(len(input_field))
