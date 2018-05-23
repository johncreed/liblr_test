#!/usr/bin/python3
import os
import sys
from os.path import join
from os import makedirs
import math
import multiprocessing as mp
home = "/home/johncreed/"
tmp = join(home, "tmp")
log = join(tmp, "all_logs")
out_path = ""
program_path = join(home, "liblr_test/train")

# reg data
data_path =  join(home, "reg")
command_param = ""
type = {0: "warm-fixP-go-C", 
        1:"warm-fixC-go-P",
        2: "No-warm-fixP-go-C"
        }
big_data_list = ['log1p.E2006.train', 'YearPredictionMSD', 'E2006.train'] 
small_data_list = [ f for f in os.listdir(data_path) if f not in big_data_list ]
def remove_dir( path ):
  names = os.listdir(path)
  for name  in names:
    if os.path.isdir(join(path, name)):
      remove_dir( path )
    else:
      os.remove(join( path, name))
  os.rmdir( path )

def clear_dir( path ):
  names = os.listdir(path)
  for name in names:
    pth = join(path, name)
    if os.path.isdir(pth):
      remove_dir(pth)
    else:
      os.remove(pth)
 
'''
# clv data
data_path = join(home, clv)
command_param = " -s 0 -C"
data_list = [ f for f in os.listdir(data_path) if f not in big_data_list ]
small_data_list = [
  "news20.binary",
  "rcv1_train.binary",
  "w8a",
  "real-sim",
  "yahoojp",
  "yahookr",
  "url_combined",
  "HIGGS",
  "kdda",
  "kddb",
  # "webspam_wc_normalized_trigram.svm",
]
'''


# Functions 
def go( file ):
    cmd = program_path + command_param + " " + join(data_path, file) + " > "+ join(out_path, file)
    print (cmd)
    os.system(cmd)

train_list = []
def choose_list():
    global train_list
    train_list = input("train big or small (B/S): ")
    if train_list == 'B':
        train_list = big_data_list
    else:
        train_list = small_data_list

def print_list_with_idx( myList ):
  idx = 0
  myList.sort()
  for item in myList:
    print("{} : {}".format(idx,item))
    idx = idx + 1

def choose_output_folder():
    global out_dir, out_path
    all_log_folders = [x for x in os.listdir(log)]
    all_log_folders.sort()
    print_list_with_idx( all_log_folders )
    choose_idx = int(input("Which folder to read log file or -1 to create new one: "))
    if choose_idx == -1:
      #out_dir = input("Which folder to store output: ")
      command_param_list = command_param.split()
      log_type_str = type[int(command_param_list[command_param_list.index('-t') + 1])]
      solver_type = str(command_param_list[command_param_list.index('-s') + 1])
      eps = str(command_param_list[command_param_list.index('-e') + 1])
      out_dir = "testlog-s{}-{}-{}".format(solver_type,eps,log_type_str)
      out_path = join(log, out_dir)
      if out_dir in os.listdir(log):
          print ("Folder exist !!!")
          cmf = input("rm all the elements(y/n) in {} ? ".format(out_path))
          if cmf == 'y':
              clear_dir(out_plog)
          elif cmf == 'n':
              sys.exit("Try again")
      else:
          makedirs(out_path)
    else:
      out_dir = all_log_folders[choose_idx]
      out_path = join(tmp, out_dir)
    print ("The log path : {}".format(out_path))
    return

def set_command_param():
  global command_param
  solver = input("solver = -s (1,2,11)")
  e = input("-e = ")
  t = input("0: (warm fixP go C) 1:(warm fixC go P) 2: (No warm fixP go C)")
  command_param = " -s {} -C -e {} -t {}".format(solver, e, t)

def __main__():
    set_command_param()
    choose_output_folder()
    choose_list()
    if train_list == big_data_list :
        my_str = ""
        count = 0
        for file in train_list:
            print (file)
            my_str = my_str + str(count) + ":" + file + " "
            count += 1
        print (my_str)
        file = train_list[int(input(my_str))]
        go(file)
    else:
        for file in train_list:
            go(file)


__main__()

