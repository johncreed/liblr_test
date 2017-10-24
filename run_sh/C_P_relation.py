#!/usr/bin/python3
import sys
import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as tic
import math

from scipy.interpolate import griddata
import numpy.ma as ma
from numpy.random import uniform, seed

import pylab as p
import mpl_toolkits.mplot3d.axes3d as p3

out_path = "/home/johncreed/liblr_test/out/"
pictures_path = "/home/johncreed/liblr_test/pictures/"
file_names = [f for f in listdir(out_path)]

def draw():
    for file_name in file_names:
        x = []
        y = []
        z = []
        with open(out_path+file_name) as f:
            for line in f:
                tmp = line.split();
                if(len(tmp) == 3 ):
                    x.append(float(tmp[0]))
                    y.append(float(tmp[1]))
                    z.append(math.log10(float(tmp[2])))
        xa = np.asarray(x)
        ya = np.asarray(y)
        za = np.asarray(z)
        # define grid.
        xi = np.linspace(-50,20,70)
        yi = np.linspace(-60,30,90)
        # grid the data.
        zi = griddata((xa, ya), za, (xi[None,:], yi[:,None]), method='cubic')
        # contour the gridded data, plotting dots at the randomly spaced data points.
        CS = plt.contour(xi,yi,zi,15,linewidths=0.5,colors='k')
        CS = plt.contourf(xi,yi,zi,15,cmap=plt.cm.jet)
        plt.colorbar() # draw colorbar
        # plot data points.
        plt.scatter(x,y,marker='.',c='white',s=5)
        plt.xlim(-50,20)
        plt.ylim(-60,30)
        plt.title(file_name)
        plt.savefig(pictures_path+file_name+".eps", format="eps", dpi=1000)
        plt.close()
        
def draw_3D():
    for file_name in file_names:
        print(file_name)
        x = []
        y = []
        z = []
        with open(out_path+file_name) as f:
            for line in f:
                tmp = line.split();
                if(tmp[0] == "log2c"):
                    x.append(float(tmp[1]))
                    y.append(float(tmp[3]))
                    z.append(math.log10(float(tmp[5])))
        xa = np.asarray(x)
        ya = np.asarray(y)
        za = np.asarray(z)

        #angle_list = [0 , 270]
        #for angle in angle_list:
        fig=p.figure()
        ax = p3.Axes3D(fig)
        ax.scatter3D(xa,ya,za)
        ax.set_xlabel('C')
        ax.set_ylabel('P')
        ax.set_zlabel('Error')
        ax.view_init(10, -30)
        fig.add_axes(ax)
        plt.savefig(pictures_path+file_name+"_"+".eps", format="eps", dpi=1000)
        plt.close()




draw_3D()
