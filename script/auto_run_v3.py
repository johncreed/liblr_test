#/usr/bin/python3
import os
import sys
import math
import multiprocessing as mp
eps_range = [ x for x in range(-50, 20) if x % 3 == 0]
eps_b = 2 ** -10
eps_e = 2 ** 10
eps = eps_b
train_big = False
data_path = "/tmp/b02701216/"
out_path = "~/liblr_test/out_p/"
program_path = "~/liblr_test/"
command_param = "train -s 11 -P -c "
big_data_list = ['log1p.E2006.train', 'YearPredictionMSD', 'E2006.train']
small_data_list = [ f for f in os.listdir(data_path) if f not in big_data_list ]

file = big_data_list[0]

def go_multi(eps):
    cmd = program_path+command_param+str(2**eps)+" "+data_path+file+" > "+out_path+file+"_"+str(eps)
    print (cmd)
    os.system(cmd)

def multi():
    p = mp.Pool(5)
    p.map(go, eps_range)
    p.close()

multi()

def go( eps , file):
    cmd = program_path+command_param+str(2**eps)+" "+data_path+file+" > "+out_path+file+"_"+str(eps)
    print (cmd)
    os.system(cmd)


def greed_eps():
    for file in small_data_list:
        for eps in eps_range:
            go(eps, file);




