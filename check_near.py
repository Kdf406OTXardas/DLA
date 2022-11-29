import numpy as np
import random as rnd
import time
from time import gmtime, strftime

def print_str(a,b):
    print(str(a),b)


rand_final=[]
random_list =[]
max=5

pore_procent = 50
fore_pore_volume = pore_procent/100*max*max
pore_volume = divmod(fore_pore_volume,1)[0]
field_test = np.zeros((max+2,max+2))
first_coords = divmod(max,2)[0]+1 ,divmod(max,2)[0]+1
print(first_coords)
list_for_check_calculate = np.array([-1, 0, 1])

random_list_land=np.array((None,None))



def land_list():
    global random_list_land
    for x in range(max):
        for y in range(max):
            random_list_land= np.vstack([random_list_land,np.array([x+1,y+1])])   

def random_list_vstack_crds(for_del):
    global random_list_land
    random_list_land = np.delete(random_list_land ,(for_del) ,axis=0)
    
def zero_point():
    global field_test
    global first_coords
    field_test[first_coords]=1

def new_point():
    global field_test
    global random_list_land
    x,y = rnd.choice(random_list_land)
    field_test[x,y] = 2
    
def rnd_list():
    for x_rand in range(max):
            for y_rand in range(max):
                random_list.append([x_rand+1,y_rand+1])

def new_check_list():
    global check_list
    global list_for_check_calculate
    check_list=np.array([0,0])
    for y in list_for_check_calculate:
        for z in list_for_check_calculate:
            # if ((y!=0)==True and (z!=0)==True)==True:
            for_new = np.array([y,z])
            check_list = np.vstack([check_list ,for_new])
    check_list=np.delete(check_list ,(0,5) ,axis=0)

print(field_test)

land_list()
random_list_vstack_crds(first_coords)
zero_point()
new_point()
# random_list_vstack_np()
print(field_test)


#/====== Можно раскоментировать и посмотреть что будет если частица находится рядом ======/
    # field_test = np.zeros((max+2,max+2))
    # field_test[divmod(max,2)[0]+1 ,divmod(max,2)[0]+1]=1
    # field_test[divmod(max,2)[0]+2 ,divmod(max,2)[0]+1]=2
    # print(field_test)

#проверка соседей
print_str(field_test,'field_test before check')

def check_neighbours(): #существует вариант, что новая частица попадет под проверку
    global check_list
    global check_field
    global field_test    
    for i in check_list:
        check_field = np.roll(field_test,(i[0],i[1]),axis = (0,1))
        check_field=field_test*check_field
        x ,y = np.where(check_field==2)
        # print_str(check_field, 'check_field')
        # print_str(field_test, 'field_test')
        field_test[x-i[0],y-i[1]]=1
        if x.size>0:
            random_list_vstack_crds([x-i[0],y-i[1]])
            new_point()
            

new_check_list()
check_neighbours()
print_str(field_test,'field_test after check')


x ,y = np.where(field_test==2)
print(x ,y)
for i in range(len(x)):
    criteria_sum = 1
    check_list_there = check_list
    while criteria_sum == 1:
        a=rnd.choice(check_list_there)
        print(a)
        print(x[-i])
        if x[-i]+a[0]>0 & x[-i]+a[0]<max+1 & y[-i]+a[1]>0 & y[-i]+a[1]<max+1:
            criteria_sum = 0
            break
        else:
            if check_list_there!=[]:
                check_list_there = np.delete(check_list_there, [a[0] ,a[1]],axis = 0)
            else:
                criteria_sum = 0
                break
    field_test[x[-i],y[-i]]=0        
    field_test[x[-i]+a[0],y[-i]+a[1]]=2
    
print_str(field_test,'field_test after move')
