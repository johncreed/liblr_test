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

home = "/home/johncreed"
tmp = join(home, "tmp")
pic_path = ""


def transposeList( myL ):
  return list(map(list, zip(*myL)))

def matchGetNext( keyword, order , myL):
  idx = 0
  while( order >= 1 ):
    try:
      idx = myL.index(keyword, int(idx))
    except ValueError:
      print( myL )
      sys.exit( "matchGetNext Fail: keyword \"{}\" not found".format(keyword) )
    order = order - 1
  if idx < len(myL) - 1:
    return myL[idx+1]
  else:
    sys.exit( "matchGetNext Fail: Invalid index range!" )

def choose_pic_folder(log_path, output_label):
    global pic_path
    pic_dir = "{}-{}".format(output_label,basename(log_path))
    if pic_dir in os.listdir(tmp):
        cmf = input("rm all the elements(y/n) ? ")
        if cmf == 'n':
            sys.exit("Retry Again")
        else:
            os.system("rm "+join(tmp, pic_dir)+"/*")
    else:
        makedirs(join(tmp, pic_dir))
    pic_path = join(tmp, pic_dir)
    print("pic_path is {}".format(pic_path))
    return

def print_list_with_idx( myList ):
  idx = 0
  myList.sort()
  for item in myList:
    print("{} : {}".format(idx,item))
    idx = idx + 1

def set_log_path():
    log_path = ""
    print("=== set_log_path ===")
    all_log_folders = [ x for x in os.listdir(tmp) ]
    all_log_folders.sort()
    print_list_with_idx( all_log_folders )
    choose_idx = int(input("Which folder to read log file: "))
    log_path = join(tmp, all_log_folders[choose_idx] )
    print("log_path is {}".format(log_path) )
    return log_path

def read_log_file(file_path):
      #(P, C, ERROR)
      cvs = [[],[],[]]
      old = [[],[],[]]
      new = [[],[],[]]
      best = [[],[],[]]
      #(Fix param, iteration, cumulative iteration)
      oldIter = [[],[],[]]
      newIter = [[],[],[]]
      with open(file_path) as f:
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
              if(tmp[0] == "Old" and tmp[2] == "Iteration:"):
                P = matchGetNext("cur_logP:", 1, tmp)
                P = 0.0 if P == "INF" else 2 ** float(P)
                iter_sum = matchGetNext("iter_sum:", 1, tmp)
                iter_sum = float(iter_sum)
                oldIter[0].append(P)
                oldIter[1].append(iter_sum)
                oldIter[2].append(sum(oldIter[1]))
              if(tmp[0] == "New" and tmp[2] == "Iteration:"):
                if( tmp[3] == "cur_logP:" ):
                  P = matchGetNext("cur_logP:", 1, tmp)
                  P = 0.0 if P == "INF" else 2 ** float(P)
                  iter_sum = matchGetNext("iter_sum:", 1, tmp)
                  iter_sum = float(iter_sum)
                  newIter[0].append(P)
                  newIter[1].append(iter_sum)
                  newIter[2].append(sum(newIter[1]))
                else:
                  C = matchGetNext("cur_logC:", 1, tmp)
                  C = 0.0 if C == "INF" else 2 ** float(C)
                  iter_sum = matchGetNext("iter_sum:", 1, tmp)
                  iter_sum = float(iter_sum)
                  newIter[0].append(C)
                  newIter[1].append(iter_sum)
                  newIter[2].append(sum(newIter[1]))
      return { "cvs" : cvs,
              "old" : old,
              "new" : new,
              "best" : best,
              "oldIter" : oldIter,
              "newIter" : newIter
              }

def logList( origList ):
  return [log(x)/log(2.0) for x in origList]

def draw_3D():
  log_path = set_log_path()
  all_file_names = [f for f in os.listdir(log_path)]
  choose_pic_folder(log_path, "[Graph-3D]")
  for file_name in all_file_names:
    print ("Do " + file_name)
    file_path = join(log_path, file_name)
    dictResult = read_log_file(file_path)
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

def draw_2D():
  log_path = set_log_path()
  all_file_names = [f for f in os.listdir(log_path)]
  choose_pic_folder(log_path, "[Graph-2D]")
  for file_name in all_file_names:
    print ("Do " + file_name)
    file_path = join(log_path, file_name)
    dictResult = read_log_file(file_path)
    fix, ax = plt.subplots()
    plt.title(file_name)
   
    # Line1 is the CV score and parameter C
    cvs = transposeList(dictResult["cvs"])
    x = [ point[1] for point in cvs if point[0] == 0.0]
    y = [ point[2] for point in cvs if point[0] == 0.0]
    line1 = ax.plot(logList(x), logList(y), 'b--o',linewidth=2, label='Warm Start')
    # Point1 is old Break
    old_break = transposeList(dictResult["old"])
    x = [ point[1] for point in old_break if point[0] == 0.0]
    y = [ point[2] for point in old_break if point[0] == 0.0]
    line2 = ax.scatter(logList(x), logList(y), color = "r", marker = 'o', s = 200, label='Old Break')
    # Point2 is new Break
    new_break = transposeList(dictResult["new"])
    x = [ point[1] for point in new_break if point[0] == 0.0]
    y = [ point[2] for point in new_break if point[0] == 0.0]
    line3 = ax.scatter(logList(x), logList(y), color = "g", marker = 'o',s = 200, label='New Break')
    
    ax.set_xlabel("Log2( C )")
    ax.set_ylabel("Log10(MSE)")
    plt.legend(scatterpoints=1)
    plt.savefig(join(pic_path,file_name+".eps"), format="eps", dpi=1000)
    plt.close()


def draw_fixP_vs_fixC():
  print ("=== Choose FixC Folder ===")
  FixC_log_path = set_log_path()
  print ("=== Choose FixP Folder ===")
  FixP_log_path = set_log_path()
  choose_pic_folder(FixC_log_path, "[Graph-FixP-vs-FixC-Cmp]")
  print (pic_path)
  all_file_names = [f for f in os.listdir(FixC_log_path)]
  for file_name in all_file_names:
    print ("Do " + file_name)
    FixC_file_path = join(FixC_log_path, file_name)
    FixP_file_path = join(FixP_log_path, file_name)
    fixCDict = read_log_file( FixC_file_path )
    fixPDict = read_log_file( FixP_file_path)

    maxCUsed = max( fixPDict["cvs"][1] )
    print( "Max C used is {}".format(maxCUsed) )

    # Create subplots with 1 rows/2 cols and share the same y-axis.
    fig, (ax1,ax2) = plt.subplots(1,2,sharey = True)
    plt.title(file_name)
    
    # Draw ax1
    bw = 0.5 # bar width
    PValue = fixPDict["newIter"][0]
    PIter = fixPDict["newIter"][1]
    PBarLoc = np.arange(len(PIter))
    fixPCulIter = fixPDict["newIter"][2][-1]
    BC1 = ax1.bar(PBarLoc, PIter, width=bw, align='center')
    # Set x ticks
    gap = int ( len(PIter) / 10.0 )
    Pxticklabels = []
    for i , x in enumerate(PValue):
      if i % gap == 0:
          Pxticklabels.append(str( round(x,2) ))
      else:
        Pxticklabels.append('')
    ax1.set_xticklabels(Pxticklabels,rotation=90)
    ax1.set_xticks(PBarLoc)
    # Set title
    ax1.set_title("Fix Parameter P : {}".format(fixPCulIter))
    # Set x label and y label
    ax1.set_xlabel("P")
    ax1.set_ylabel("Iteration")

    # Draw ax2
    # Set the first greater bar in ax2 to be red
    bw = 0.5 # bar width
    CValue = fixCDict["newIter"][0]
    CIter = fixCDict["newIter"][1]
    CBarLoc = np.arange(len(CIter))
    # Find the index of the first greater then maxCUsed in CValue
    idx = len(CValue) - 1
    color = ['b' for x in CValue]
    for i, x in enumerate(CValue):
      if x > maxCUsed:
        color[i] = 'r'
        idx = i
        break
    print ( "len {} idx {}".format(len(CValue), idx))
    #
    idxDraw = idx + 1
    fixCCulIter = fixCDict["newIter"][2][idx]
    BC2 = ax2.bar(CBarLoc[:idxDraw], CIter[:idxDraw], width=bw, align='center', color=color[:idxDraw])
    # Set x ticks
    gap = int ( idxDraw / 10.0 )
    Cxticklabels = []
    for i , x in enumerate(CValue):
      if i == idx:
        break
      if i % gap == 0:
          Cxticklabels.append(str( round(log(x)/log(2.0),2) ))
      else:
        Cxticklabels.append('')
    print (Cxticklabels)
    ax2.set_xticklabels(Cxticklabels,rotation=90)
    ax2.set_xticks(CBarLoc[:idxDraw])
    # Set title
    ax2.set_title("Fix Parameter C : {}".format(fixCCulIter))
    # Set x label and y label
    ax2.set_xlabel("log2(C)")
    ax2.set_ylabel("Iteration")
    
    fig.suptitle(file_name, fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    fig.savefig(join(pic_path,file_name)+".eps", format="eps", dpi=1000)
    plt.close()
   
    """
    objects = tuple(P_labels + C_labels)
    len1 = len(P_labels)
    len2 = len(C_labels)
    wd = 3
    x = [a * wd for a in range(len1)]
    x = x + [(a + len1 + 1) * wd for a in range(len2)]
    x = np.array(x)

    plt.figure(figsize=(20,10))
    barlist = plt.bar(x, y_val, width=wd, align='center', alpha=1)
    
    red_bar_index = len1 + int(max_c - min_c)
    print(max_c - min_c)
    if len(barlist) >= red_bar_index + 1:
        barlist[red_bar_index].set_color('r')
    plt.xticks(x, objects)
    plt.xlabel('Parameter Fix P or Fix C')
    plt.title(file_name + " " + str(max_c))
    plt.savefig(pic_path+file_name+".eps", format="eps", dpi=1000)
    plt.close()
    """


# Define which draw picture name and corresponded function
gDict = {"[Graph-3D]" : draw_3D ,
         "[Graph-2D]" : draw_2D ,
         "[Graph-FixP-vs-FixC-Cmp]" : draw_fixP_vs_fixC
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
