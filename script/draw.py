#!/usr/bin/python3
import sys
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as tic
from matplotlib import rcParams
#rcParams.update({'figure.autolayout': True})
import math
from math import log10 as log
from scipy.interpolate import griddata
import numpy.ma as ma
from numpy.random import uniform, seed
import pylab
import mpl_toolkits.mplot3d.axes3d as p3
from os import makedirs
from os.path import join, basename
import collections 

all_logs = "log"
all_graphs = "pic"

ext = lambda x: x[x.rfind('.')+1:]
name = lambda x: x[x.rfind('/')+1:]
trim = lambda x: x[:x.rfind('.')]

def escape_keyword( myS ):
  escapeWordList = ["_", "."]
  for escapeWord in escapeWordList:
    myS = myS.replace( escapeWord, "-")
  return myS

def transposeList( myL ):
  return list(map(list, zip(*myL)))

def matchVal( keyword, order , myL):
  idx = 0
  while( order >= 1 ):
    try:
      idx = myL.index(keyword, int(idx))
    except ValueError:
      print( myL )
      sys.exit( "matchVal Fail: keyword \"{}\" not found".format(keyword) )
    order = order - 1
    idx = idx + 1
  if idx < len(myL):
    return myL[idx]
  else:
    sys.exit( "matchVal Fail: Invalid index range!" )

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

def choose_pic_folder(log_path, pic_type):
    pic_dir = "{}-{}".format(pic_type,basename(log_path))
    pic_path = join(all_graphs, pic_dir)
    print([x for x in os.listdir(all_graphs)])
    if pic_dir not in [x for x in os.listdir(all_graphs)]:
        os.makedirs(pic_path)
    else:
        clear_dir(pic_path)
    print("pic_path is {}".format(pic_path))
    return pic_path

def print_list_with_idx( myList ):
  idx = 0
  myList.sort()
  for item in myList:
    print("{} : {}".format(idx,item))
    idx = idx + 1

def logToNoLog( val ):
  infList = ["INF", "-INF", "inf", "-inf"]
  if val not in infList:
    return float(pow( 2.0, float(val)))
  else:
    return 0.0

def set_log_path(folder_name):
    log_path = ""
    print("=== Choose {} ===".format(folder_name))
    all_log_folders = [ x for x in os.listdir(all_logs) ]
    all_log_folders.sort()
    print_list_with_idx( all_log_folders )
    choose_idx = int(input("Which folder to read log file: "))
    log_path = join(all_logs, all_log_folders[choose_idx] )
    print("log_path is {}".format(log_path) )
    return log_path

def read_log_file(file_path):
      #(P, C, ERROR)
      best = [[],[],[]]
      #(iter_sum, culmulative iter_sum, P, C)
      iterSum = [[],[],[],[]]
      with open(file_path) as f:
          for line in f:
              tmp = line.split();
              if(tmp[0] == "iter_sum:"):
                iterSum[0].append( int(matchVal("iter_sum:", 1, tmp)))
                iterSum[1].append(sum(iterSum[0]))
                iterSum[2].append( logToNoLog( matchVal("log2P:", 1, tmp)))
                iterSum[3].append( logToNoLog( matchVal("log2C:", 1, tmp)))
              if(tmp[:2] == "Best log2C:".split()):
                best[1].append( logToNoLog( matchVal("log2C:", 1, tmp)))
                best[2].append( float( matchVal("Acc:", 1, tmp)))
      return {
              "best" : best,
              "iterSum" : iterSum
              }

def log2List( origList ):
  ret = []
  for x in origList:
      if x != 0.0:
          ret.append(log(x) / log(2.0))
      else:
          ret.append(-30)
  return ret

def acc_table():
    print("s0old")
    s0old_dir = set_log_path("s0old")
    print("s0new")
    s0new_dir = set_log_path("s0new")
    print("s2old")
    s2old_dir = set_log_path("s2old")
    print("s2new")
    s2new_dir = set_log_path("s2new")

    pic_path = choose_pic_folder(ext(s2new_dir), "[Table-Acc-Comparison]")
    f = open(join(pic_path,'acc-table'), 'w')
    file_list = [ trim(trim(f)) for f in os.listdir(s0old_dir)]
    f.write(" & {} & {} & {} & {} \\\\ \n".format(s0old_dir, s0new_dir, s2old_dir, s2new_dir))
    e = ext(s0old_dir)
    for name in file_list:
        s0old = join(s0old_dir, "{}.{}.s0old".format(name, e))
        s0new = join(s0new_dir, "{}.{}.s0new".format(name, e))
        s2old = join(s2old_dir, "{}.{}.s2old".format(name, e))
        s2new = join(s2new_dir, "{}.{}.s2new".format(name, e))

        s0old_D = read_log_file(s0old)
        s0new_D = read_log_file(s0new)
        s2old_D = read_log_file(s2old)
        s2new_D = read_log_file(s2new)

        s0old_best = s0old_D['best'][2][0]
        s0new_best = s0new_D['best'][2][0]
        s2old_best = s2old_D['best'][2][0]
        s2new_best = s2new_D['best'][2][0]

        f.write("{} & {} & {} \\\\ \n".format( name, round(s0new_best*100 / s0old_best,2) , round(s2new_best * 100 / s2old_best, 2))) 
       # f.write(" {} & {} & {} & {} & {} \\\\ \n".format(name, s0old_best, s0new_best, s2old_best, s2new_best))

def cls_iter_table():
    print("s0old")
    s0old_dir = set_log_path("s0old")
    print("s0new")
    s0new_dir = set_log_path("s0new")
    print("s2old")
    s2old_dir = set_log_path("s2old")
    print("s2new")
    s2new_dir = set_log_path("s2new")

    pic_path = choose_pic_folder(ext(s2new_dir), "[Table-cls-iter-Comparison]")
    f = open(join(pic_path,'cls-iter-table'), 'w')
    file_list = [ trim(trim(f)) for f in os.listdir(s0old_dir)]
    e = ext(s0old_dir)
    f.write(" & {} & {} & {} & {} \\\\ \n".format(s0old_dir, s0new_dir, s2old_dir, s2new_dir))
    for name in file_list:
        s0old = join(s0old_dir, "{}.{}.s0old".format(name, e))
        s0new = join(s0new_dir, "{}.{}.s0new".format(name, e))
        s2old = join(s2old_dir, "{}.{}.s2old".format(name, e))
        s2new = join(s2new_dir, "{}.{}.s2new".format(name, e))

        s0old_D = read_log_file(s0old)
        s0new_D = read_log_file(s0new)
        s2old_D = read_log_file(s2old)
        s2new_D = read_log_file(s2new)

        s0old_iter =sum( s0old_D['iterSum'][0])
        s0new_iter =sum( s0new_D['iterSum'][0])
        s2old_iter =sum( s2old_D['iterSum'][0])
        s2new_iter =sum( s2new_D['iterSum'][0])


        f.write(" {} & {} & {} \\\\ \n".format(name, round ( 100 * s0new_iter / s0old_iter,2), round(100 * s2new_iter/ s2old_iter, 2)) )
        #f.write(" {} & {} & {} & {} & {} \\\\ \n".format(name, s0old_iter, s0new_iter, s2old_iter, s2new_iter))

# Define which draw picture name and corresponded function
gDict = {
         "[Table-acc-Comparison]": acc_table, 
         "[Table-iter-Comparison]": cls_iter_table ,
         }

def choose_graph_type():
  gList = list(gDict)
  print("=== Choose Graph Type ===")
  print_list_with_idx(gList)
  choose_idx = int(input("Enter index number : "))
  graph_type = gList[choose_idx]
  print ( "Graph {}".format(graph_type))
  return graph_type


def __main__():
  gFunc = gDict[choose_graph_type()]
  gFunc()

__main__()
