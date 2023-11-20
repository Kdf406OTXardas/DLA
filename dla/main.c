#include <stdio.h>
#include <stdlib.h>
#include <time.h>


int main() {
    // Degree of space 3d
    int SPACE_VALUE = 3;
    // Input EDGE_FIELD with +2
    int EDGE_FIELD = 6;
    int END_POROSITY = 20; // Input porosity [0:100]
    int NOW_POROSITY = 0; // porosity in the calculation process
    int FOR_NOW_POROSITY = (EDGE_FIELD - 2) * (EDGE_FIELD - 2) * (EDGE_FIELD - 2);
    int MAX_COORD = EDGE_FIELD - 2;
    // Field size for calculating
    int SIZE_FIELD = EDGE_FIELD;

    int i;

// ======== Full size of the calculation field ========
    for (i = 1; i < SPACE_VALUE; i++){
        SIZE_FIELD = SIZE_FIELD * EDGE_FIELD;
    }

// ======== Filling calculating field by 0 ========
    int calculating_field[SIZE_FIELD];
    for (i = 0;  i < SIZE_FIELD; i++){
        calculating_field[i] = 0;
    }

// ======== First immobilized point ========
    int immobilize_points = 1;
    int counter_immobilize_points = 0;
    int middle_index = EDGE_FIELD / 2 + EDGE_FIELD * (EDGE_FIELD / 2 + EDGE_FIELD * EDGE_FIELD / 2);
    calculating_field[middle_index] = 1;

// ======== New random point ========
    time_t t;
    srand((unsigned) time(&t));

    int counter_in_for = 2;
    int check_counter = 0; // For the limit of calculating new coordinates
    int random_point; // Index of new point coordinates
    int prev_coords_point;
    int mobile_points = 0;
    int z; // Save z
    int y; // Save y
    int x; // Save x
    int counter_render = 0; // To display the field correctly

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
//                printf("%d z\n", z);

                // Y-coord new point
                y = (middle_index - z * EDGE_FIELD * EDGE_FIELD) / EDGE_FIELD + (rand() % 3 - 1) * immobilize_points;
                if (y < 1) {
                    y = 1;
                } else if (y > MAX_COORD) {
                    y = MAX_COORD;
                }
//                printf("%d y\n", y);

                // X-coord new point
                x = middle_index - EDGE_FIELD * (y + EDGE_FIELD * z) + (rand() % 3 - 1) * immobilize_points;
                if (x < 1) {
                    x = 1;
                } else if (x > MAX_COORD) {
                    x = MAX_COORD;
                }
//                printf("%d x\n", x);

                // Index of new point
                random_point = x + EDGE_FIELD * (y + EDGE_FIELD * z);

                    if ((calculating_field[random_point] == 0) || (check_counter == 120)) {
                        printf("if_2 random point %d\n", random_point);
                        printf("if_2 middle_index %d\n", middle_index);
                        break;
                    } else {
                        printf("if_3 %d\n", random_point + middle_index);
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
        printf("random_point after for %d\n", random_point);
        printf("immobilize_points after for %d\n", immobilize_points);
        calculating_field[random_point] = 2;
//        printf("%d %d %d z, y, x\n", z, y, x);

        //======= Print field START =======
        for (i = 0; i < SIZE_FIELD; i++) {
            if (counter_render % EDGE_FIELD == 0) {
                printf("\n");
            }
            if (counter_render % (EDGE_FIELD * EDGE_FIELD) == 0) {
                printf("\nLayer_Z: %d \n", (i + 1) / EDGE_FIELD / EDGE_FIELD);
            }
            counter_render += 1;
            if (calculating_field[i] == 1) {
                printf("X ");
            } else if (calculating_field[i] == 2) {
                printf("M ");
            } else {
                printf("o ");
            }
            //======= Print field END =======
        }

// ======== Check neighbours ========
        // Y level +0; condition written for CUDA
        for (i = 0; i < 1; i++) {
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
            printf("\n%d z\n", z);

            // Y of moving point
            y = y + rand() % 3 - 1;
            if (y < 1) {
                y = 1;
            } else if (y > MAX_COORD) {
                y = MAX_COORD;
            }
            printf("%d y\n", y);

            // New X of moving point
            x = x + rand() % 3 - 1;
            if (x < 1) {
                x = 1;
            } else if (x > MAX_COORD) {
                x = MAX_COORD;
            }
            printf("%d x\n", x);

            // Index of new point
            calculating_field[random_point] = 0;
            printf("random_point before 0 %d x\n", random_point);
            random_point = x + EDGE_FIELD * (y + EDGE_FIELD * z);
            printf("new random_point %d x\n", random_point);
            calculating_field[random_point] = 2;
            printf("random_point after 0 %d x\n", random_point);
        }
    }
//!!! ======= End main algorithm =======

    //======= Print field START =======
    for (i = 0;  i < SIZE_FIELD; i++){
        if (counter_render % EDGE_FIELD == 0){
            printf("\n");
        }
        if (counter_render % (EDGE_FIELD * EDGE_FIELD) == 0){
            printf("\nLayer_Z: %d \n", (i + 1) / EDGE_FIELD / EDGE_FIELD);
        }
        counter_render += 1;
        if (calculating_field[i]== 1){
            printf("X ");
        } else if (calculating_field[i] == 2){
            printf("M ");
        } else {
            printf("o ");
        }
        //======= Print field END =======
    }


    return NOW_POROSITY;
}