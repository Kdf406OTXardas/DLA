import numpy as np
import random as rnd
from time import gmtime, strftime

def print_str(a,b):
    print(str(a),b)

def zero_point():
    global points
    num_point = 1
    points = np.array([num_point ,1 ,divmod(max,2)[0]+1 ,divmod(max,2)[0]+1])
    print(points)


def new_point():
    global next_point
    
    try:
        rows_points
    except NameError:
        rows_points = 2
    else:
        rows_points +=1
    
    new_row = np.array([rows_points ,2 ,rnd.randrange(1,max+1) ,rnd.randrange(1,max+1)])
    # next_point = points
    next_point = np.vstack([points ,new_row])
    print_str(next_point ,'after')
    
rand_final=[]
random_list =[]
max=4

pore_procent = 50
fore_pore_volume = pore_procent/100*max*max
pore_volume=divmod(fore_pore_volume,1)[0]

field_test = np.zeros((max+2,max+2))
list_for_check_calculate = np.array([-1, 0, 1])


def rnd_list():
    for x_rand in range(max):
            for y_rand in range(max):
                random_list.append([x_rand+1,y_rand+1])
    # print_str(random_list,'random_list')

def points_to_field():
    field_test.fill(0)
    for i in range(len(next_point)):
        field_test[next_point[i][2]][next_point[i][3]]= next_point[i][1]

# print_str(next_point,"""next_point""")
print_str(next_point[1][2],"""коорд""")
# print_str(next_point[1][3],"""коорд""")
rnd_list()
zero_point()
new_point()
points_to_field()
# print_str(field_test,"""field_test""")


def checking_matrix():
    global check_down_right
    global check_up_right
    global check_down_left
    global check_up_left
    check_up_left = np.roll(field_test,(1,-1),axis = (0,1))
    check_up_right = np.roll(field_test,(1,1),axis = (0,1))
    check_down_left = np.roll(field_test,(-1,-1),axis = (0,1))
    check_down_right = np.roll(field_test,(-1,1),axis = (0,1))
    
checking_matrix()
print(field_test)
print(check_up_left)
print(check_up_right)
print(check_down_left)
print(check_down_right)