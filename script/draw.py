#!/usr/bin/python3
import sys
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as tic
import math
from math import log10 as log
from scipy.interpolate import griddata
import numpy.ma as ma
from numpy.random import uniform, seed
import pylab
import mpl_toolkits.mplot3d.axes3d as p3
from os import makedirs
from os.path import join, basename

home = "/home/johncreed"
tmp = join(home, "tmp")
out_path = ""
pic_path = ""
graph_type = ""



def result_dict(out_path,file_name):
      #(P, C, ERROR)
      cvs = [[],[],[]]
      old = [[],[],[]]
      new = [[],[],[]]
      best = [[],[],[]]
      with open(join(out_path,file_name)) as f:
          for line in f:
              tmp = line.split();
              if(tmp[0] == "log2P="):
                if tmp[1] == "INF":
                  cvs[0].append(0)
                else:
                  cvs[0].append(2**float(tmp[1]))
                cvs[1].append(2**float(tmp[3]))
                cvs[2].append(float(tmp[5]))
              if(tmp[0] == "Old" and tmp[-2] == "MSE="):
                if tmp[3] == "INF":
                  old[0].append(0)
                else:
                  old[0].append(2**float(tmp[3]))
                old[1].append(2**float(tmp[5]))
                old[2].append(float(tmp[-1]))
              if(tmp[0] == "New" and tmp[-2] == "MSE="):
                if tmp[3] == "INF":
                  new[0].append(0)
                else:
                  new[0].append(2**float(tmp[3]))
                new[1].append(2**float(tmp[5]))
                new[2].append(float(tmp[-1]))
              if(tmp[0] == "Best"):
                if tmp[3] == "INF":
                  best[0].append(0.0)
                else:
                  best[0].append(2**float(tmp[3]))
                best[1].append(2**float(tmp[7]))
                best[2].append(float(tmp[-1]))
      
      return { "cvs" : cvs,
              "old" : old,
              "new" : new,
              "best" : best
              }
      # To ndarray
      for i in range(3):
        cvs[i] = np.asarray(cvs[i])
        old[i] = np.asarray(old[i])
        new[i] = np.asarray(new[i])
        best[i] = np.asarray(best[i])

def logList( origList ):
  return [log(x)/log(2.0) for x in origList]

def draw_3D(file_name):
      dictResult = result_dict(out_path, file_name)
      print(type(dictResult["best"][2]))
      #angle_list = [0,10,20,40,50,60,70,80,90,100,110,120,130,140,140,160,170,180] # 0 ~ 90
      angle_list = [20,80]
      cnt = "a"
      for angle in angle_list:
          fig=pylab.figure()
          ax = p3.Axes3D(fig)
          tle = file_name
          pylab.title(tle)
          ax.scatter3D(dictResult["best"][0],logList(dictResult["best"][1]),dictResult["best"][2], s =  [300],c = "yellow", marker = 'o')
          ax.scatter3D(dictResult["cvs"][0],logList(dictResult["cvs"][1]),dictResult["cvs"][2], s=[10] ,c = "black",marker = 'o')
          ax.scatter3D(dictResult["new"][0],logList(dictResult["new"][1]),dictResult["new"][2], s =  [80],c = "green", marker = 'o')
          ax.scatter3D(dictResult["old"][0],logList(dictResult["old"][1]),dictResult["old"][2], s =  [80],c = "Red", marker = 'o')
          ax.set_xlabel('P')
          ax.set_ylabel('C')
          ax.set_zlabel('Error')
          ax.view_init(30, angle)
          fig.add_axes(ax)
          #plt.savefig(join(pic_path,file_name+"-"+str(angle)+".eps"), format="eps", dpi=1000)
          plt.savefig(join(pic_path,file_name+"-"+cnt+".eps"), format="eps", dpi=1000)
          cnt += "a"
          plt.close()


def do_gif(file_name):
  cmd = "convert -delay 20 -loop 0 " + "{}.eps".format(join(pic_path, "*")) + " " + "{}.gif".format(join(pic_path,file_name))
  os.system(cmd)
  cmd = "rm " + join(pic_path, "*.eps")
  os.system(cmd)

def draw_2D(file_name):
  dictResult = result_dict(out_path, file_name)
  fix, ax = plt.subplots()
  plt.title(file_name)
 
  # Line1 is the CV score and parameter C
  cvs = list(map(list, zip(*dictResult["cvs"])))
  x = [ point[1] for point in cvs if point[0] == 0.0]
  y = [ point[2] for point in cvs if point[0] == 0.0]
  line1 = ax.plot(logList(x), logList(y), 'b--o',linewidth=2, label='Warm Start')
  # Point1 is old Break
  old_break = list(map(list, zip(*dictResult["old"])))
  x = [ point[1] for point in old_break if point[0] == 0.0]
  y = [ point[2] for point in old_break if point[0] == 0.0]
  line2 = ax.scatter(logList(x), logList(y), color = "r", marker = 'o', s = 200, label='Old Break')
  # Point2 is new Break
  new_break = list(map(list, zip(*dictResult["new"])))
  x = [ point[1] for point in new_break if point[0] == 0.0]
  y = [ point[2] for point in new_break if point[0] == 0.0]
  line3 = ax.scatter(logList(x), logList(y), color = "g", marker = 'o',s = 200, label='New Break')
  
  ax.set_xlabel("Log2( C )")
  ax.set_ylabel("Log10(MSE)")
  plt.legend(scatterpoints=1)
  plt.savefig(join(pic_path,file_name+".eps"), format="eps", dpi=1000)
  plt.close()


gDict = {"3D-Graph" : draw_3D ,"2D-Graph" : draw_2D }
def choose_graph_type():
  idx = 0
  for gType in gDict:
    print( "{} : {}".format(idx, gType))
    idx = idx + 1
  which_one = int(input("Enter index number : "))
  global graph_type
  graph_type = list(gDict)[which_one]

def choose_path_folder():
    global out_path
    os.system("ls {}".format(tmp))
    out_dir = input("Which folder to read out file: ")
    if out_dir not in os.listdir(tmp):
      sys.exit('out dir not found!')
    else:
      out_path = join(tmp, out_dir)

def choose_pic_folder():
    global pic_path
    pic_dir = "{}-{}".format(graph_type,basename(out_path))
    if pic_dir in os.listdir(tmp):
        cmf = input("rm all the elements(y/n) ? ")
        if cmf == 'n':
            sys.exit("Retry Again")
        else:
            os.system("rm "+join(tmp, pic_dir)+"/*")
    else:
        makedirs(join(tmp, pic_dir))
    pic_path = join(tmp, pic_dir)
    return

def __main__():
  choose_graph_type()
  choose_path_folder()
  choose_pic_folder()
  gFunc = gDict[graph_type]
  print ( "Graph {}".format(graph_type))
  print (out_path)
  print (pic_path)
  file_names = [f for f in os.listdir(out_path)]
  for file in file_names:
      print ("Do " + file)
      gFunc(file)

__main__()
