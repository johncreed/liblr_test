#!/usr/bin/python3
import os
import sys
import math
import multiprocessing as mp
eps_range = [ x for x in range(-10, 10)]
eps_b = 2 ** -10
eps_e = 2 ** 10
eps = eps_b
train_big = False
data_path = "/tmp2/b02701216/"
out_path = "~/liblr_test/out2/"
program_path = "~/liblr_test/"
command_param = "train -s 11 -C -p "
big_data_list = ['log1p.E2006.train', 'YearPredictionMSD', 'E2006.train']
small_data_list = [ f for f in os.listdir(data_path) if f not in big_data_list ]

"""
def go(f):
    cmd = program_path+command_param+str(eps)+" "+data_path+f+" > "+out_path+f+"_"+str(math.log(eps, 2))
    print (cmd)
    os.system(cmd)


def train():
    if(train_big):
        pool = mp.Pool(5)
        pool.map(go,big_data_list)
        pool.close()
    else:
        #go(small_data_list)
        pool = mp.Pool(5)
        pool.map(go,small_data_list)
        pool.close()

while eps_b <= eps < eps_e:
    train()
    print (eps , "finished")
    eps *= 2

"""


file = big_data_list[ int(sys.argv[1]) ]
eps_range = [x for x in range(int(sys.argv[2]), int(sys.argv[3]) )]
for i in eps_range:
    print (i)

def go( eps ):
    cmd = program_path+command_param+str(2**eps)+" "+data_path+file+" > "+out_path+file+"_"+str(eps)
    print (cmd)
    os.system(cmd)


def multi():
    p = mp.Pool(5)
    p.map(go, eps_range)
    p.close()

multi()

"""

if (len(sys.argv) > 1):
    if(sys.argv[1] == 1):
        train_big = True
    else:
        train_big = False

if(train_big):
    for f in big_data_list:
        file = f
        pool = mp.Pool(5)
        pool.map(go,eps_range)
        pool.close()
        for x in eps_range:
            go(x)
else:
    for f in small_data_list:
        file = f
        pool = mp.Pool(5)
        pool.map(go,eps_range)
        pool.close()
        for x in eps_range:
            go(x)

"""
