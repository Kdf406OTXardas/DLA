import numpy as np
import random as rnd
import time
from time import gmtime, strftime

def print_str(a,b):
    print(str(a),b)

def zero_point():
    global points
    num_point = 1
    points = np.array([num_point ,1 ,divmod(max,2)[0]+1 ,divmod(max,2)[0]+1])
    points = np.vstack([points ,np.array([num_point ,1 ,divmod(max,2)[0]+2 ,divmod(max,2)[0]+1])])
    print(points)

def new_point():
    global next_point
    global new_row
    
    try:
        rows_points
    except NameError:
        rows_points = 2
    else:
        rows_points +=1
    
    new_row = np.array([rows_points ,2 ,divmod(max,2)[0]+1 ,divmod(max,2)[0]])
    # new_row = np.array([rows_points ,2 ,rnd.randrange(1,max+1) ,rnd.randrange(1,max+1)])

    next_point = points
    
    next_point = np.vstack([next_point ,new_row])
    next_point = np.vstack([next_point ,np.array([rows_points ,2 ,divmod(max,2)[0]+2 ,divmod(max,2)[0]])])
    
    print_str(next_point ,'after')
    
rand_final=[]
random_list =[]
max=5

pore_procent = 50
fore_pore_volume = pore_procent/100*max*max
pore_volume=divmod(fore_pore_volume,1)[0]

field_test = np.zeros((max+2,max+2))

def rnd_list():
    for x_rand in range(max):
            for y_rand in range(max):
                random_list.append([x_rand+1,y_rand+1])
    # print_str(random_list,'random_list')

def points_to_field():
    field_test.fill(0)
    for i in range(len(next_point)):
        field_test[next_point[i][2]][next_point[i][3]]= next_point[i][1]

rnd_list()
zero_point()
new_point()
print_str(len(next_point),'len')
points_to_field()

def CreateCheckList():
    global CheckList
    list_for_check_calculate = np.array([-1, 0, 1])
    CheckList = np.array([0,0])
    for y in list_for_check_calculate:
        for z in list_for_check_calculate:
            # if ((y!=0)==True and (z!=0)==True)==True:
            for_new = np.array([y,z])
            CheckList = np.vstack([CheckList ,for_new])
    CheckList=np.delete(CheckList ,(0) ,axis=0)
    CheckList=np.delete(CheckList ,(4) ,axis=0)

def CheckAndChange():
    for i in CheckList:
        CheckField = np.roll(field_test,(i[0],i[1]),axis = (0,1))
        CheckField = field_test*CheckField
        x,y = np.where(CheckField==2)
        field_test[x-i[0],y-i[1]]=1
        
print_str(field_test,'field_test before')
CreateCheckList()
CheckAndChange()
print_str(field_test,'field_test after')