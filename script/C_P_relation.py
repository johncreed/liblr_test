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

home = "/home/johncreed/"
tmp = home + "tmp/"
out_path = ""
pic_path = ""

def choose_output_folder():
    global pic_path, out_path
    out_dir = input("Which folder to read out file: ")
    if out_dir not in os.listdir(tmp):
      sys.exit('out dir not found!')
    else:
      out_path = tmp + out_dir + "/"

    pic_dir = input("Which folder to store pic: ")
    if pic_dir in os.listdir(tmp):
        cmf = input("Replace the elements(y/n) ? ")
        if cmf == 'n':
            sys.exit("Retry Again")
    else:
        cmd = "mkdir " + tmp + pic_dir
        os.system(cmd)
    pic_path = tmp + pic_dir + "/"
    return

def draw_3D(file_name):
      c = []
      p = []
      err = []
      old = [[],[],[]]
      new = [[],[],[]]
      tle = ""
      min_P = rel_diff  = 0.0
      with open(out_path+file_name) as f:
          for line in f:
              tmp = line.split();
              if(tmp[0] == "log2p="):
                p.append(float(tmp[1]))
                c.append(float(tmp[3]))
                err.append(math.log10(float(tmp[5])))
              if(tmp[0] == "Old"):
                old[0].append(float(tmp[3]))
                old[1].append(float(tmp[5]))
                old[2].append(math.log10(float(tmp[-1])))
              if(tmp[0] == "New"):
                new[0].append(float(tmp[3]))
                new[1].append(float(tmp[5]))
                new[2].append(math.log10(float(tmp[-1])))
              
      c = np.asarray(c)
      p = np.asarray(p)
      err = np.asarray(err)
      for i in range(3):
        old[i] = np.asarray(old[i])
        new[i] = np.asarray(new[i])
      
      angle_list = [0,90] # 0 ~ 90
      for angle in angle_list:
          fig=pylab.figure()
          ax = p3.Axes3D(fig)
          pylab.title(tle)
          ax.scatter3D(p,c,err, s=[1] ,c = "b",marker = '.')
          ax.scatter3D(old[0],old[1],old[2], s = [80],c = "red", marker = 'o')
          ax.scatter3D(new[0],new[1],new[2], s =  [40],c = "green", marker = 'o')
          ax.set_xlabel('P')
          ax.set_ylabel('C')
          ax.set_zlabel('Error')
          ax.view_init(0, angle)
          fig.add_axes(ax)
          plt.savefig(pic_path+file_name+"_"+str(angle)+".eps", format="eps", dpi=1000)
          plt.close()


def draw_3D_log_P(file_name):
      cvs = [[],[],[]]  # (p, c, err)
      old = [[],[],[]]
      new = [[],[],[]]
      tle = ""
      min_P = rel_diff  = 0.0
      with open(out_path+file_name) as f:
          for line in f:
              tmp = line.split();
              if(tmp[0] == "log2p="):
                cvs[0].append(float(tmp[1]))
                cvs[1].append(float(tmp[3]))
                cvs[2].append(math.log10(float(tmp[5])))
              if(tmp[0] == "Old"):
                old[0].append(float(tmp[3]))
                old[1].append(float(tmp[5]))
                old[2].append(math.log10(float(tmp[-1])))
              if(tmp[0] == "New"):
                new[0].append(float(tmp[3]))
                new[1].append(float(tmp[5]))
                new[2].append(math.log10(float(tmp[-1])))
      # turn cvs[0] to log scale
      for i in range(cvs[0]):
        cvs[0][i] = math.log(cvs[0][i], 2.0)
      for i in range(old[0]):
        old[0][i] = math.log(old[0][i], 2.0)
      for i in range(new[0]):
        new[0][i] = math.log(new[0][i], 2.0)

      # To ndarray
      for i in range(3):
        cvs[i] = np.asarray(cvs[i])
        old[i] = np.asarray(old[i])
        new[i] = np.asarray(new[i])
      
      angle_list = [0,90] # 0 ~ 90
      for angle in angle_list:
          fig=pylab.figure()
          ax = p3.Axes3D(fig)
          pylab.title(tle)
          ax.scatter3D(cvs[0],cvs[1],cvs[2], s=[1] ,c = "b",marker = '.')
          ax.scatter3D(old[0],old[1],old[2], s = [80],c = "red", marker = 'o')
          ax.scatter3D(new[0],new[1],new[2], s =  [40],c = "green", marker = 'o')
          ax.set_xlabel('P')
          ax.set_ylabel('C')
          ax.set_zlabel('Error')
          ax.view_init(0, angle)
          fig.add_axes(ax)
          plt.savefig(pic_path+file_name+"_"+str(angle)+".eps", format="eps", dpi=1000)
          plt.close()


def __main__():
  choose_output_folder()
  file_names = [f for f in os.listdir(out_path)]
  logOrNot = input('Log Or not (y/n): ')
  for file in file_names:
      print ("Do " + file)
      if( logOrNot == 'y' ):
        draw_3D_log_P(file)
      else:
        draw_3D(file)

__main__()
