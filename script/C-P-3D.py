#!/usr/bin/python3
import sys
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as tic
import math
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
    pic_dir = "3D-Graph-{}".format(basename(out_path))
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


def result_dict(out_path,file_name):
      cvs = [[],[],[]]  # (p, c, err)
      old = [[],[],[]]
      new = [[],[],[]]
      best = [[],[],[]]
      min_P = rel_diff  = 0.0
      with open(join(out_path,file_name)) as f:
          for line in f:
              tmp = line.split();
              if(tmp[0] == "log2p="):
                if tmp[1] == "INF":
                  cvs[0].append(0)
                else:
                  cvs[0].append(2**float(tmp[1]))
                cvs[1].append(float(tmp[3]))
                cvs[2].append(math.log10(float(tmp[5])))
              if(tmp[0] == "Old"):
                if tmp[3] == "INF":
                  old[0].append(0)
                else:
                  old[0].append(2**float(tmp[3]))
                old[1].append(float(tmp[5]))
                old[2].append(math.log10(float(tmp[-1])))
              if(tmp[0] == "New"):
                if tmp[3] == "INF":
                  new[0].append(0)
                else:
                  new[0].append(2**float(tmp[3]))
                new[1].append(float(tmp[5]))
                new[2].append(math.log10(float(tmp[-1])))
              if(tmp[0] == "Best"):
                if tmp[7] == "-inf":
                  best[0].append(0.0)
                else:
                  best[0].append(2**float(tmp[7]))
                best[1].append(float(tmp[3]))
                best[2].append(math.log10(float(tmp[-1])))
      
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



def draw_3D(file_name):
      dictResult = result_dict(out_path, file_name)
      #angle_list = [0,10,20,40,50,60,70,80,90,100,110,120,130,140,140,160,170,180] # 0 ~ 90
      angle_list = [20,80]
      cnt = "a"
      for angle in angle_list:
          fig=pylab.figure()
          ax = p3.Axes3D(fig)
          tle = file_name
          pylab.title(tle)
          ax.scatter3D(dictResult["best"][0],dictResult["best"][1],dictResult["best"][2], s =  [300],c = "yellow", marker = 'o')
          ax.scatter3D(dictResult["cvs"][0],dictResult["cvs"][1],dictResult["cvs"][2], s=[10] ,c = "black",marker = 'o')
          ax.scatter3D(dictResult["new"][0],dictResult["new"][1],dictResult["new"][2], s =  [80],c = "green", marker = 'o')
          ax.scatter3D(dictResult["old"][0],dictResult["old"][1],dictResult["old"][2], s =  [80],c = "Red", marker = 'o')
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

def __main__():
  choose_path_folder()
  choose_pic_folder()
  print (out_path)
  print (pic_path)
  file_names = [f for f in os.listdir(out_path)]
  for file in file_names:
      print ("Do " + file)
      draw_3D(file)

__main__()
