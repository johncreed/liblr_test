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
data_path = "/home/johncreed/data/reg/"
out_path = "/home/johncreed/liblr_test/out/"
program_path = "/home/johncreed/liblr_test/"
command_param = "train -s 11 -C"
big_data_list = ['log1p.E2006.train', 'YearPredictionMSD', 'E2006.train']
small_data_list = [ f for f in os.listdir(data_path) if f not in big_data_list ]

file = "123"

def go():
    cmd = program_path+command_param+" "+data_path+file+" > "+out_path+file+"_"+str(eps)
    print (cmd)
    os.system(cmd)

if (len(sys.argv) > 1):
    if(sys.argv[1] == 1):
        train_big = True
    else:
        train_big = False

if(train_big):
    for f in big_data_list:
        file = f
        go()
else:
    for f in small_data_list:
        file = f
        go()
