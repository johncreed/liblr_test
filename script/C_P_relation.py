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
        b_C = []
        b_P = []
        C = []
        P = []
        b_err = []
        err = []
        tle = ""
        min_P = rel_diff  = 0.0
        with open(out_path+file_name) as f:
            for line in f:
                tmp = line.split();
                if(tmp[0] == "log2c"):
                    x.append(float(tmp[1]))
                    y.append(float(tmp[3]))
                    z.append(math.log10(float(tmp[5])))
                if(tmp[0] == "Bound"):
                    b_C.append(float(tmp[4]))
                    b_P.append(float(tmp[8]))
                    b_err.append(math.log10(float(tmp[-1])))
                if(tmp[0] == "Best"):
                    C.append(float(tmp[3]))
                    P.append(float(tmp[7]))
                    err.append(math.log10(float(tmp[-1])))
                if(tmp[0] == "min_P"):
                    tle = tmp[0] + "=" + tmp[-1]
                    min_P = float(tmp[-1])
                if(tmp[0] == "Relative"):
                    rel_diff = float(tmp[-1])
        xa = np.asarray(x)
        ya = np.asarray(y)
        za = np.asarray(z)
        b_Ca = np.asarray(b_C)
        b_Pa = np.asarray(b_P)
        b_erra = np.asarray(b_err)
        Ca = np.asarray(C)
        Pa = np.asarray(P)
        erra = np.asarray(err)
        print(b_Ca, b_Pa, b_erra)
        print(Ca, Pa, erra)
        tle = tle + " " + "Best: " + str(10 ** err[0])[0:10] + " Bound: " + str(10**b_err[0])[0:10] + "\n" + " Relative diff: " + str(rel_diff)[0:10]

        b_l_x = np.linspace(-20,40, 100)
        b_l_y = np.linspace(min_P, min_P, 100)
        b_l_z = np.linspace(err[0], err[0], 100)

        angle_list = [0 , 270]
        for angle in angle_list:
            fig=p.figure()
            ax = p3.Axes3D(fig)
            p.title(tle)
            ax.scatter3D(xa,ya,za, s=[1] ,c = "b",marker = '.')
            ax.scatter3D(b_Ca,b_Pa,b_erra, s = [80],c = "red", marker = 'o')
            ax.scatter3D(Ca,Pa, erra, s =  [40],c = "green", marker = 'o')
            ax.plot(b_l_x, b_l_y, b_l_z)
            ax.set_xlabel('C')
            ax.set_ylabel('P')
            ax.set_zlabel('Error')
            ax.view_init(10, angle)
            fig.add_axes(ax)
            plt.savefig(pictures_path+file_name+"_"+str(angle)+".eps", format="eps", dpi=1000)
            plt.close()




draw_3D()
