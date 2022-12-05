import numpy as np
# import cupy as cupy
import random as rnd
import time
from time import gmtime, strftime
from settings import *


# Эта строчка нигде не используется
# target_pore_cells = TARGET_POROSITY / 100 * FIELD_SIZE * FIELD_SIZE
# Эта строчка нигде не используется. Зачем она?
# pore_volume = divmod(target_pore_cells, 1)[0]
# Зачем здесь деление с остатком? Достаточно обычного же
# first_coords = divmod(FIELD_SIZE, 2)[0] + 1, divmod(FIELD_SIZE, 2)[0] + 1



rand_final = []

random_list_land = np.array((None, None))

check_list = []

check_field = []
check_list_there = []


def render_field(field):
    dim = field.shape
    for y in range(dim[1]):
        row = ''
        for x in range(dim [0]):
            if field[x, y] == 0:
                row += '.'
            elif field[x, y] == 1:
                row += 'K'
            elif field[x, y] == 2:
                row += 'O'
        print(row)
    print()

def print_str(a, b):
    print(str(a), b)


# Зачем так сложно? Почему просто нельзя рандомно расположить сразу на массиве?
# Уйти от глобальной переменной
def land_list():
    global random_list_land
    for x in range(FIELD_SIZE):
        for y in range(FIELD_SIZE):
            random_list_land = np.vstack([random_list_land, np.array([x+1, y+1])])


def random_list_vstack_crds(for_del):
    global random_list_land
    random_list_land = np.delete(random_list_land, for_del, axis=0)

# Такие вещи лучше не глобально ставить, а давать аргументами. Это делает функцию более обобщенной


def place_cluster(field, place_coords):
    field[place_coords] = 1


def new_point(field, points_set):
    x, y = rnd.choice(points_set)
    field[x, y] = 2

# Эта функция не используется


def rnd_list(random_list):
    for x_rand in range(FIELD_SIZE):
        for y_rand in range(FIELD_SIZE):
            random_list.append([x_rand+1, y_rand+1])


def new_check_list(list_for_check_calculate):
    global check_list
    check_list = np.array([0, 0])
    for y in list_for_check_calculate:
        for z in list_for_check_calculate:
            # if ((y!=0)==True and (z!=0)==True)==True:
            for_new = np.array([y, z])
            check_list = np.vstack([check_list, for_new])
    check_list = np.delete(check_list, (0, 5), axis=0)

# /====== Можно раскоментировать и посмотреть что будет если частица находится рядом ======/
    # field_test = np.zeros((max+2,max+2))
    # field_test[divmod(max,2)[0]+1 ,divmod(max,2)[0]+1]=1
    # field_test[divmod(max,2)[0]+2 ,divmod(max,2)[0]+1]=2
    # print(field_test)


# существует вариант, что новая частица попадет под проверку


def check_neighbours(field):
    global check_list
    global check_field
    global random_list_land

    for i in check_list:
        check_field = np.roll(field, (i[0], i[1]), axis=(0, 1))
        check_field=field*check_field
        x, y = np.where(check_field == 2)
        # print_str(check_field, 'check_field')
        # print_str(field_test, 'field_test')
        field[x-i[0], y-i[1]] = 1
        if x.size > 0:
            random_list_vstack_crds([x-i[0], y-i[1]])
            new_point(field, random_list_land)

# Чтобы код не представлял из себя мешанину и было легче ориентироваться, делают точку входа в виде функции main
# Сюда помещают весь исполняемый код


def main():
    global list_for_check_calculate
    global check_list_there
    global random_list_land


    land_list()
    #print(random_list_land)
    list_for_check_calculate = np.array([-1, 0, 1])
    field_test = np.zeros((FIELD_SIZE + 2, FIELD_SIZE + 2))
    render_field(field_test)
    #print(field_test)
    initial_cluster_coords = FIELD_SIZE // 2 + 1, FIELD_SIZE // 2 + 1
    random_list_vstack_crds(initial_cluster_coords)
    place_cluster(field_test, initial_cluster_coords)

    new_point(field_test, random_list_land)
    render_field(field_test)
    #random_list_vstack_np()
    #print(field_test)
    # проверка соседей
    #print_str(field_test, 'field_test before check')

    print('Поле до')
    render_field(field_test)

    new_check_list(list_for_check_calculate)
    check_neighbours(field_test)
    #print_str(field_test, 'field_test after check')
    print('Поле после проверки')
    render_field(field_test)

    x, y = np.where(field_test == 2)
    #print(x, y)
    for i in range(len(x)):
        criteria_sum = 1
        check_list_there = check_list
        while criteria_sum == 1:
            a = rnd.choice(check_list_there)
            #print(a)
            #print(x[-i])
            if x[-i] + a[0] > 0 & x[-i] + a[0] < FIELD_SIZE + 1 & y[-i] + a[1] > 0 & y[-i] + a[1] < FIELD_SIZE + 1:
                criteria_sum = 0
                break
            else:
                if check_list_there != []:
                    check_list_there = np.delete(check_list_there, [a[0], a[1]], axis=0)
                else:
                    criteria_sum = 0
                    break
        field_test[x[-i], y[-i]] = 0
        field_test[x[-i] + a[0], y[-i] + a[1]] = 2

    #print_str(field_test, 'field_test after move')
    print('Поле после движения')

    render_field(field_test)


main()

