import numpy as np
from pycuda import driver, compiler, gpuarray, tools
import pycuda.autoinit

SPACE_VALUE = 3
EDGE_FIELD = 10
END_POROSITY = 10
FOR_NOW_POROSITY = int((EDGE_FIELD - 2) * (EDGE_FIELD - 2) * (EDGE_FIELD - 2))
MAX_COORD = EDGE_FIELD - 2
SIZE_FIELD = EDGE_FIELD

input_field = np.zeros(SIZE_FIELD ** SPACE_VALUE).astype(np.float32)

middle_index = int(EDGE_FIELD / 2 + EDGE_FIELD * (EDGE_FIELD / 2 + EDGE_FIELD * EDGE_FIELD / 2))

input_field[middle_index] = float(1)


kernel_code_template = """ 

#include <stdio.h>
#include <stdlib.h>
#include <time.h>


__global__ void CalculatingDLA(float *calculating_field, int EDGE_FIELD, int END_POROSITY,
                                int FOR_NOW_POROSITY, int MAX_COORD, int SIZE_FIELD,
                                int middle_index)
{
    int i;

    time_t t;
    srand((unsigned) time(&t));
    
    int immobilize_points = 1;
    int counter_in_for = 2;
    int counter_immobilize_points = 0;
    int check_counter = 0; // For the limit of calculating new coordinates
    int random_point; // Index of new point coordinates
    int mobile_points = 0;
    int z; // Save z
    int y; // Save y
    int x; // Save x
    
    int NOW_POROSITY = 0; // porosity in the calculation process

    
    
    
//!!! ======= Start main algorithm =======
    while (NOW_POROSITY < END_POROSITY) {
        
        // Calculating of coordinates of a new point
        if (mobile_points == 0) {
            for (i = 0; i < counter_in_for; i++) {

                // Z-coord new point
                z = middle_index / (EDGE_FIELD * EDGE_FIELD) + (rand() % 3 - 1) * immobilize_points;
                if (z < 1) {
                    z = 1;
                } else if (z > MAX_COORD) {
                    z = MAX_COORD;
                }

                // Y-coord new point
                y = (middle_index - z * EDGE_FIELD * EDGE_FIELD) / EDGE_FIELD + (rand() % 3 - 1) * immobilize_points;
                if (y < 1) {
                    y = 1;
                } else if (y > MAX_COORD) {
                    y = MAX_COORD;
                }

                // X-coord new point
                x = middle_index - EDGE_FIELD * (y + EDGE_FIELD * z) + (rand() % 3 - 1) * immobilize_points;
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
                        i = 0;
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

        // Y level -1 /(like +0, but with "- EDGE_FIELD * EDGE_FIELD"; condition written for CUDA
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

        // Y level +1 /(like +0, but with "+ EDGE_FIELD * EDGE_FIELD"; condition written for CUDA
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
            z = z + rand() % 3 - 1;
            if (z < 1) {
                z = 1;
            } else if (z > MAX_COORD) {
                z = MAX_COORD;
            }

            // Y of moving point
            y = y + rand() % 3 - 1;
            if (y < 1) {
                y = 1;
            } else if (y > MAX_COORD) {
                y = MAX_COORD;
            }

            // New X of moving point
            x = x + rand() % 3 - 1;
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

field_gpu = gpuarray.to_gpu(input_field)
gpu_output = gpuarray.empty(SIZE_FIELD ** SPACE_VALUE, np.float32)


kernel_code = kernel_code_template

mod = compiler.SourceModule(kernel_code)

DLA_output = mod.get_function("MatrixMulKernel")

DLA_output(
    field_gpu, EDGE_FIELD, END_POROSITY, FOR_NOW_POROSITY, FOR_NOW_POROSITY, FOR_NOW_POROSITY, middle_index,
    gpu_output,
    block=(SIZE_FIELD ** SPACE_VALUE, 1))

print(input_field)
print(len(input_field))
