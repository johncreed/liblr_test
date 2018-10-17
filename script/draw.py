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

home = "/home/johncreed"
tmp = join(home, "tmp")
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
      cvs = [[],[],[]]
      old = [[],[],[]]
      new = [[],[],[]]
      best = [[],[],[]]
      #(Fix param, iteration, cumulative iteration)
      oldIter = [[],[],[]]
      newIter = [[],[],[]]
      #(iter_sum, culmulative iter_sum, P, C)
      iterSum = [[],[],[],[]]
      with open(file_path) as f:
          for line in f:
              tmp = line.split();
              if(tmp[0] == "iter_sum:"):
                  if(tmp[2] ==  "log2P:"):
                    iterSum[0].append( int(matchVal("iter_sum:", 1, tmp)))
                    iterSum[1].append(sum(iterSum[0]))
                    iterSum[2].append( logToNoLog( matchVal("log2P:", 1, tmp)))
                    iterSum[3].append( logToNoLog( matchVal("log2C:", 1, tmp)))
                  elif(tmp[2] == "log2C"):
                    iterSum[0].append( int(matchVal("iter_sum:", 1, tmp)))
                    iterSum[1].append(sum(iterSum[0]))
                    iterSum[3].append( logToNoLog( matchVal("log2C:", 1, tmp)))
              if(tmp[0] == "log2P:"):
                cvs[0].append( logToNoLog( matchVal("log2P:", 1, tmp)))
                cvs[1].append( logToNoLog( matchVal("log2C:", 1, tmp)))
                cvs[2].append( float(matchVal("MSE:", 1, tmp)))
              if("Old Break log2P:".split() == tmp[:3]):
                old[0].append( logToNoLog( matchVal("log2P:", 1, tmp)))
                old[1].append( logToNoLog( matchVal("log2C:", 1, tmp)))
                old[2].append(float( matchVal("MSE:", 1, tmp)))
              if("New Break log2P:".split() == tmp[:3]):
                new[0].append( logToNoLog( matchVal("log2P:", 1, tmp)))
                new[1].append( logToNoLog( matchVal("log2C:", 1, tmp)))
                new[2].append(float( matchVal("MSE:", 1, tmp)))
              if("Old Break Iteration:".split() == tmp[:3]):
                oldIter[0].append( logToNoLog( matchVal("log2P:", 1, tmp)))
                oldIter[1].append( float( matchVal("iter_sum:", 1, tmp)))
                oldIter[2].append(sum(oldIter[1]))
              if("New Break Iteration:".split() == tmp[:3]):
                if( tmp[3] == "log2P:" ):
                  newIter[0].append( logToNoLog( matchVal("log2P:", 1, tmp)))
                  newIter[1].append( float( matchVal("iter_sum:", 1, tmp)))
                  newIter[2].append(sum(newIter[1]))
                elif( tmp[3] == "log2C:" ):
                  newIter[0].append( logToNoLog( matchVal("log2C:", 1, tmp)))
                  newIter[1].append( float( matchVal("iter_sum:", 1, tmp)))
                  newIter[2].append(sum(newIter[1]))
              if(tmp[:2] == "Best log2P:".split()):
                best[0].append( logToNoLog( matchVal("log2P:", 1, tmp)))
                best[1].append( logToNoLog( matchVal("log2C:", 1, tmp)))
                best[2].append( float( matchVal("MSE:", 1, tmp)))
              if(tmp[:2] == "Best log2C:".split()):
                best[1].append( logToNoLog( matchVal("log2C:", 1, tmp)))
                best[2].append( float( matchVal("Acc:", 1, tmp)))
      return { "cvs" : cvs,
              "old" : old,
              "new" : new,
              "best" : best,
              "oldIter" : oldIter,
              "newIter" : newIter,
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

def draw_warm_vs_noWarm():
  print ("=== warm file ===")
  warm_log_path = set_log_path()
  print ("=== noWarm file ===")
  noWarm_log_path = set_log_path()
  getExt = lambda x : x[x.rfind('.')+1:]
  trimExt = lambda y : y[:y.rfind('.')]
  pic_path = choose_pic_folder("{}-{}".format(getExt(warm_log_path),getExt(noWarm_log_path)), "[Graph-warm-vs-noWarm]")
  all_file_names = [trimExt(trimExt(f)) for f in os.listdir(noWarm_log_path)]
  all_file_names2 = [trimExt(f) for f in os.listdir(warm_log_path)]
  if len(all_file_names) > len(all_file_names2):
    all_file_names = all_file_names2
  f = open(join(pic_path, "warm-vs-noWarm-table"), 'w')
  tbl_cnt = 1
  for file_name in all_file_names:
    print ("Do {}".format(file_name))
    warmDict = read_log_file( join(warm_log_path, "{}.fixPgoC".format(file_name)))
    noWarmDict = read_log_file( join(noWarm_log_path, "{}.fixPgoC.noWarm".format(file_name)) )
    warmIterSum = warmDict["iterSum"]
    noWarmIterSum = noWarmDict["iterSum"]
    warmIterSum_T = transposeList(warmIterSum)
    noWarmIterSum_T = transposeList(noWarmIterSum)


    # Get break C list for each fix P
    breakCDict = {}
    new = warmDict["new"]# [ [P], [C], [MSE] ]
    new_T = transposeList(new)
    for i in range(len(new_T)):
        breakCDict[new_T[i][0]] = new_T[i][1]

    # Get iteration culmulative for each fix P
    warmCulIterDict = {}
    noWarmCulIterDict = {}
    
    curP = -1
    for inst in warmIterSum_T :
      cnt = inst[0]
      P = inst[2]
      C = inst[3]
      if curP != P :
        curP = P
        culIter = 0
        warmCulIterDict[curP] = {}
      culIter = culIter + cnt
      warmCulIterDict[curP][C] = culIter
   
    curP = -1
    for inst in noWarmIterSum_T :
      cnt = inst[0]
      P = inst[2]
      C = inst[3]
      if curP != P :
        curP = P
        culIter = 0
        noWarmCulIterDict[curP] = {}
      culIter = culIter + cnt
      noWarmCulIterDict[curP][C] = culIter
    
    tmp = join(pic_path, file_name)
    os.mkdir(tmp)
    
    warmTotalIter = 0
    noWarmTotalIter = 0
    for P in warmCulIterDict:
      warmCulList = sorted( warmCulIterDict[P].items() )
      noWarmCulList = sorted( noWarmCulIterDict[P].items())[:len(warmCulList)]
     
      # Data to draw
      warmCList = [ x[0] for x in warmCulList]
      warmList = [ x[1] for x in warmCulList]
      noWarmCList = [ x[0] for x in noWarmCulList]
      noWarmList = [ x[1] for x in noWarmCulList]

      # Trim data untill break C occur:
      breakC = breakCDict[P]
      try:
          idx = warmCList.index(breakC)
      except ValueError:
          sys.exit("breakC not found.")

      idx = idx + 1
      warmCList = warmCList[:idx]
      warmList = warmList[:idx]
      noWarmCList = noWarmCList[:idx]
      noWarmList = noWarmList[:idx]
      
      # Check C list are equal
      compare = lambda x, y: collections.Counter(x) == collections.Counter(y)
      if not compare( warmCList, noWarmCList ):
        print (warmCList)
        print (noWarmCList)
        sys.exit( "C list are not equal.")

      fig, ax = plt.subplots(1,1)
      ax.plot( log2List(warmCList), warmList, "bo-", label="warm" )
      ax.plot( log2List(noWarmCList), noWarmList, "ro-", label="noWarm" )
      ax.set_xlabel("Log2(C)")
      ax.set_ylabel("Iteration")
      ax.legend(loc="upper left")
      fig.suptitle("{} with p = {}".format(file_name,str(round(P,3))), fontsize=18)
      ax.set_title("Iteration : {} (Warm), {} (noWarm)".format(warmList[-1], noWarmList[-1]), fontsize=12)
      fig.savefig(join(tmp,escape_keyword(file_name)+"-P-{}.eps".format(str(round(P,3)).replace(".","-"))), format="eps", dpi=1000)
      plt.close()
      warmTotalIter = warmTotalIter + warmList[-1]
      noWarmTotalIter = noWarmTotalIter + noWarmList[-1]

    if tbl_cnt % 2 :
      f.write("{} & {} & {} & \n".format(escape_keyword(file_name), str(warmTotalIter), str(noWarmTotalIter)))
    else:
      f.write("{} & {} & {} \\\\ \n".format(escape_keyword(file_name), str(warmTotalIter), str(noWarmTotalIter)))
    tbl_cnt = tbl_cnt + 1


def draw_3D():
  log_path = set_log_path()
  all_file_names = [f for f in os.listdir(log_path)]
  pic_path = choose_pic_folder(log_path, "[Graph-3D]")
  f = open(join(pic_path, "best-eps-table"), 'w')
  tbl_cnt = 1
  for file_name in all_file_names:
    print ("Do " + file_name)
    file_path = join(log_path, file_name)
    dictResult = read_log_file(file_path)
    if tbl_cnt % 2 :
      f.write("{} & ${}$ & ".format(escape_keyword(file_name), str(round(dictResult["best"][0][0], 3))))
    else :
      f.write("{} & ${}$ \\\\ \n".format(escape_keyword(file_name), str(round(dictResult["best"][0][0], 3))))
    tbl_cnt = tbl_cnt + 1

    angle_list = [20,80]
    cnt = "a"
    for angle in angle_list:
        fig=pylab.figure()
        ax = p3.Axes3D(fig)
        tle = file_name
        pylab.title(tle)
        ax.scatter3D(dictResult["best"][0],log2List(dictResult["best"][1]),log2List(dictResult["best"][2]), s =  [300],c = "yellow", marker = 'o')
        ax.scatter3D(dictResult["cvs"][0],log2List(dictResult["cvs"][1]), log2List(dictResult["cvs"][2]), s=[10] ,c = "black",marker = 'o')
        ax.scatter3D(dictResult["new"][0],log2List(dictResult["new"][1]), log2List(dictResult["new"][2]), s =  [80],c = "green", marker = 'o')
        ax.scatter3D(dictResult["old"][0],log2List(dictResult["old"][1]), log2List(dictResult["old"][2]), s =  [80],c = "Red", marker = 'o')
        ax.set_xlabel('P')
        ax.set_ylabel('log2(C)')
        ax.set_zlabel('log2(MSE)')
        ax.view_init(30, angle)
        fig.add_axes(ax)
        #plt.savefig(join(pic_path,file_name+"-"+str(angle)+".eps"), format="eps", dpi=1000)
        plt.savefig(join(pic_path,escape_keyword(file_name)+"-"+cnt+".eps"), format="eps", dpi=1000)
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
  pic_path = choose_pic_folder(log_path, "[Graph-2D]")
  for file_name in all_file_names:
    print ("Do " + file_name)
    file_path = join(log_path, file_name)
    dictResult = read_log_file(file_path)
   
    # Line1 is the CV score and parameter C
    cvs = transposeList(dictResult["cvs"])
    p_set = set(dictResult["cvs"][0])
    for p in list(p_set)[::10]:
        fix, ax = plt.subplots()
        x = [ point[1] for point in cvs if point[0] == p]
        y = [ point[2] for point in cvs if point[0] == p]
        line1 = ax.plot(log2List(x), log2List(y), 'b--o',linewidth=2, label='Warm Start')
        # Point1 is old Break
        old_break = transposeList(dictResult["old"])
        x = [ point[1] for point in old_break if point[0] == p]
        y = [ point[2] for point in old_break if point[0] == p]
        line2 = ax.scatter(log2List(x), log2List(y), color = "r", marker = 'o', s = 200, label='Old Break')
        # Point2 is new Break
        new_break = transposeList(dictResult["new"])
        x = [ point[1] for point in new_break if point[0] == p]
        y = [ point[2] for point in new_break if point[0] == p]
        line3 = ax.scatter(log2List(x), log2List(y), color = "g", marker = 'o',s = 200, label='New Break')
        
        ax.set_xlabel("Log2( C )")
        ax.set_ylabel("Log2(MSE)")
        plt.legend(scatterpoints=1)
        plt.title("{} p={}".format(file_name, round(p,3)))
        plt.savefig(join(pic_path,escape_keyword("{}.{}".format(file_name, round(p,3)))+".eps"), format="eps", dpi=1000)
        plt.close()


def draw_fixP_vs_fixC():
  print ("=== Choose FixC Folder ===")
  FixC_log_path = set_log_path()
  print ("=== Choose FixP Folder ===")
  FixP_log_path = set_log_path()
  getext = lambda x: x[x.rfind('.')+1:]
  basename = lambda x : x[x.rfind('/')+1:]
  pic_path = choose_pic_folder("{}-{}".format(basename(FixC_log_path),basename(FixP_log_path)), "[Graph-FixP-vs-FixC-Cmp]")
  file_names = [f[:f.rfind('.')] for f in os.listdir(FixC_log_path)]
  f = open(join(pic_path, 'fixP-vs-fixC-table') ,'w')
  tbl_cnt = 1
  for file_name in file_names:
    print ("Do " + file_name)
    FixC_file_path = join(FixC_log_path, "{}.fixCgoP".format(file_name))
    FixP_file_path = join(FixP_log_path, "{}.fixPgoC".format(file_name))
    fixCDict = read_log_file( FixC_file_path )
    fixPDict = read_log_file( FixP_file_path )

    maxCUsed = max( fixPDict["new"][1] )
    #print ( "Max C used is {}".format(maxCUsed) )

    # Create subplots with 1 rows/2 cols and share the same y-axis.
    fig, (ax1,ax2) = plt.subplots(1,2,sharey = True)
    plt.title(file_name)
    
    # Draw ax1
    bw = 0.5 # bar width
    PValue = fixPDict["newIter"][0]
    PIter = fixPDict["newIter"][1]
    PBarLoc = np.arange(len(PValue))
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
    ax1.set_title("Fix P Total Iteration : {}".format(int(fixPCulIter)))
    # Set x label and y label
    ax1.set_xlabel("P")
    ax1.set_ylabel("Iteration")

    # Draw ax2
    # Set the first greater bar in ax2 to be red
    bw = 0.5 # bar width
    CValue = fixCDict["newIter"][0]
    CIter = fixCDict["newIter"][1]
    CBarLoc = np.arange(len(CIter))
    # Find the index of maxCUsed in CValue
    color = ['b' for x in CValue]
    try:
        idx = CValue.index(maxCUsed)
        color[idx] = 'r'
    except ValueError:
        print("max C not found.")
    fixCCulIter = fixCDict["newIter"][2][idx]
    BC2 = ax2.bar(CBarLoc[:idx+1], CIter[:idx+1], width=bw, align='center', color=color[:idx+1])
    # Set x ticks
    gap = int ( (idx+1) / 10.0 )
    Cxticklabels = []
    for i , x in enumerate(CValue):
      if i == idx:
        break
      if i % gap == 0:
          Cxticklabels.append(str( round(log(x)/log(2.0),2) ))
      else:
        Cxticklabels.append('')
    #print (Cxticklabels)
    ax2.set_xticklabels(Cxticklabels,rotation=90)
    ax2.set_xticks(CBarLoc[:idx+1])
    # Set title
    ax2.set_title("Fix C Total Iteration: {}".format(int(fixCCulIter)))
    # Set x label and y label
    ax2.set_xlabel("log2(C)")
    ax2.set_ylabel("Iteration")
    
    fig.suptitle(file_name, fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(join(pic_path,escape_keyword(file_name))+".eps", format="eps", dpi=1000)
    plt.close()
    
    if tbl_cnt % 2:
      f.write("{} & {} & {} & ".format( escape_keyword(file_name), int(fixPCulIter), int(fixCCulIter)))
    else:
      f.write("{} & {} & {} \\\\ \n".format( escape_keyword(file_name), int(fixPCulIter), int(fixCCulIter)))
    tbl_cnt = tbl_cnt + 1 

def draw_linear_vs_log():
  linear_log_path = set_log_path("linear")
  log_log_path = set_log_path("log")
  pic_path = choose_pic_folder("{}-{}".format(name(linear_log_path),name(log_log_path)), "[Graph-linear-vs-log]")
  file_names = [trim(f) for f in os.listdir(linear_log_path)]
  for file_name in file_names:
    print ("Do " + file_name)
    linear_file_path = join(linear_log_path, "{}.PClinear".format(file_name))
    log_file_path = join(log_log_path, "{}.PClog".format(file_name))
    linearDict = read_log_file( linear_file_path )
    logDict = read_log_file( log_file_path )

    angle_list = [20,80]
    cnt = "a"
    for angle in angle_list:
        fig=pylab.figure()
        ax = p3.Axes3D(fig)
        tle = file_name
        #pylab.title(tle)
        ax.scatter3D(linearDict["cvs"][0],log2List(linearDict["cvs"][1]), log2List(linearDict["cvs"][2]), s=[10] ,c = "black",marker = 'o')
        ax.scatter3D(logDict["cvs"][0],log2List(logDict["cvs"][1]), log2List(logDict["cvs"][2]), s=[10] ,c = "red",marker = 'o')
        ax.set_xlabel(r'$\epsilon $', fontsize=13)
        ax.set_ylabel('log2(C)', fontsize=13)
        ax.set_zlabel('log2(MSE)', fontsize=13)
        ax.view_init(30, angle)
        fig.add_axes(ax)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        for t in ax.zaxis.get_major_ticks(): t.label.set_fontsize(13)
        plt.savefig(join(pic_path, "{}-{}.eps".format(escape_keyword(file_name), cnt)), format="eps", dpi=1000)
        cnt += "a"
        plt.close()

    angle_list = [20,80]
    cnt = "a"
    for angle in angle_list:
        fig=pylab.figure()
        ax = p3.Axes3D(fig)
        tle = file_name
        #pylab.title(tle)
        ax.scatter3D(log2List(linearDict["cvs"][0]),log2List(linearDict["cvs"][1]), log2List(linearDict["cvs"][2]), s=[10] ,c = "black",marker = 'o')
        ax.scatter3D(log2List(logDict["cvs"][0]),log2List(logDict["cvs"][1]), log2List(logDict["cvs"][2]), s=[10] ,c = "red",marker = 'o')
        ax.set_xlabel(r'$ log2(\epsilon )$', fontsize=13)
        ax.set_ylabel('log2(C)', fontsize=13)
        ax.set_zlabel('log2(MSE)', fontsize=13)
        ax.view_init(30, angle)
        fig.add_axes(ax)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        for t in ax.zaxis.get_major_ticks(): t.label.set_fontsize(13)
        plt.savefig(join(pic_path, "{}-{}-logP.eps".format(escape_keyword(file_name), cnt)), format="eps", dpi=1000)
        cnt += "a"
        plt.close()

def mse_table():
    nowarm_dir = set_log_path("full-nowarm")
    CPnew_dir = set_log_path("CPnew")
    PCold_dir = set_log_path("PCold")
    PCnew_dir = set_log_path("PCnew")
    PCfull_dir = set_log_path("linear")
    
    pic_path = choose_pic_folder(ext(PCfull_dir), "[Table-MSE-Comparison]")
    f = open(join(pic_path,'mse-table'), 'w')
    f.write("{} & {} & {} & {} & {}\\\\ \n".format(nowarm_dir,CPnew_dir, PCold_dir, PCnew_dir, PCfull_dir))
    file_list = [trim(f) for f in os.listdir(nowarm_dir)]
    file_list.sort()
    for name in file_list:
        name = trim(name)
        nowarm = join(nowarm_dir, "{}.1e-4.full-nowarm".format(name))
        CPnew = join(CPnew_dir, "{}.1e-4.CPnew".format(name))
        PCold = join(PCold_dir, "{}.1e-4.PCold".format(name))
        PCnew = join(PCnew_dir, "{}.1e-4.PCnew".format(name))
        PCfull = join(PCfull_dir, "{}.1e-4.PClinear".format(name))

        nowarm_D = read_log_file(nowarm)
        CPnew_D = read_log_file(CPnew)
        PCold_D = read_log_file(PCold)
        PCnew_D = read_log_file(PCnew)
        PCfull_D = read_log_file(PCfull)

        nowarm_best = min(nowarm_D['best'][2])
        CPnew_best = CPnew_D['best'][2][0] / nowarm_best
        PCold_best = PCold_D['best'][2][0]  / nowarm_best
        PCnew_best = PCnew_D['best'][2][0]  / nowarm_best
        PCfull_best = PCfull_D['best'][2][0] / nowarm_best

        f.write(" {} & {} & {} & {} & {} \\\\ \n".format( name ,round(CPnew_best, 2), round(PCold_best, 2), round(PCnew_best, 2), round(PCfull_best, 2)))
        #f.write(" {} & {} & {} & {} & {} \\\\ \n".format(name, nowarm_best, CPnew_best, PCold_best, PCnew_best))



def iter_table():
    nowarm_dir = set_log_path("full-nowarm")
    CPnew_dir = set_log_path("CPnew")
    PCold_dir = set_log_path("PCold")
    PCnew_dir = set_log_path("PCnew")
    PCfull_dir = set_log_path("linear")
    
    pic_path = choose_pic_folder(ext(PCnew_dir), "[Table-iter-Comparison]")
    f = open(join(pic_path,'iter-table'), 'w')
    f.write("{} & {} & {} & {}& {} \\\\ \n".format(nowarm_dir,CPnew_dir, PCold_dir, PCnew_dir, PCfull_dir))
    file_list = [trim(trim(f)) for f in os.listdir(nowarm_dir)]
    file_list.sort()
    for name in file_list:
        nowarm = join(nowarm_dir, "{}.1e-4.full-nowarm".format(name))
        CPnew = join(CPnew_dir, "{}.1e-4.CPnew".format(name))
        PCold = join(PCold_dir, "{}.1e-4.PCold".format(name))
        PCnew = join(PCnew_dir, "{}.1e-4.PCnew".format(name))
        PCfull = join(PCfull_dir, "{}.1e-4.PClinear".format(name))

        nowarm_D = read_log_file(nowarm)
        CPnew_D = read_log_file(CPnew)
        PCold_D = read_log_file(PCold)
        PCnew_D = read_log_file(PCnew)
        PCfull_D = read_log_file(PCfull)

        nowarm_iter = sum(nowarm_D['iterSum'][0])
        CPnew_iter = sum(CPnew_D['iterSum'][0]) / nowarm_iter
        PCold_iter = sum(PCold_D['iterSum'][0]) / nowarm_iter
        PCnew_iter = sum(PCnew_D['iterSum'][0]) / nowarm_iter
        PCfull_iter = sum(PCfull_D['iterSum'][0]) / nowarm_iter

        f.write(" {}  & {} & {} & {} & {}\\\\ \n".format(name, round(CPnew_iter,2), round(PCold_iter,2), round(PCnew_iter,2), round(PCfull_iter, 2)))

def acc_table():
    print("s0old")
    s0old_dir = set_log_path()
    print("s0new")
    s0new_dir = set_log_path()
    print("s2old")
    s2old_dir = set_log_path()
    print("s2new")
    s2new_dir = set_log_path()
    
    pic_path = choose_pic_folder(ext(s2new_dir), "[Table-Acc-Comparison]")
    f = open(join(pic_path,'acc-table'), 'w')
    file_list = [trim(f) for f in os.listdir(s0old_dir)]
    for name in file_list:
        s0old = join(s0old_dir, "{}.s0old".format(name))
        s0new = join(s0new_dir, "{}.s0new".format(name))
        s2old = join(s2old_dir, "{}.s2old".format(name))
        s2new = join(s2new_dir, "{}.s2new".format(name))

        s0old_D = read_log_file(s0old)
        s0new_D = read_log_file(s0new)
        s2old_D = read_log_file(s2old)
        s2new_D = read_log_file(s2new)

        s0old_best = s0old_D['best'][2][0]
        s0new_best = s0new_D['best'][2][0]
        s2old_best = s2old_D['best'][2][0]
        s2new_best = s2new_D['best'][2][0]

        f.write("{} & {} & {} \\\\ \n".format( name, round(s0new_best*100 / s0old_best,2) , round(s2new_best * 100 / s2old_best, 2))) 

#        f.write(" {} & {} & {} & {} & {} \\\\ \n".format(name, s0old_best, s0new_best, s2old_best, s2new_best))

def cls_iter_table():
    print("s0old")
    s0old_dir = set_log_path()
    print("s0new")
    s0new_dir = set_log_path()
    print("s2old")
    s2old_dir = set_log_path()
    print("s2new")
    s2new_dir = set_log_path()
    
    pic_path = choose_pic_folder(ext(s2new_dir), "[Table-cls-iter-Comparison]")
    f = open(join(pic_path,'cls-iter-table'), 'w')
    file_list = [trim(f) for f in os.listdir(s0old_dir)]
    for name in file_list:
        s0old = join(s0old_dir, "{}.s0old".format(name))
        s0new = join(s0new_dir, "{}.s0new".format(name))
        s2old = join(s2old_dir, "{}.s2old".format(name))
        s2new = join(s2new_dir, "{}.s2new".format(name))

        s0old_D = read_log_file(s0old)
        s0new_D = read_log_file(s0new)
        s2old_D = read_log_file(s2old)
        s2new_D = read_log_file(s2new)

        s0old_iter =sum( s0old_D['iterSum'][0])
        s0new_iter =sum( s0new_D['iterSum'][0])
        s2old_iter =sum( s2old_D['iterSum'][0])
        s2new_iter =sum( s2new_D['iterSum'][0])


        f.write(" {} & {} & {} \\\\ \n".format(name, round ( 100 * s0new_iter / s0old_iter,2), round(100 * s2new_iter/ s2old_iter, 2)) )

#        f.write(" {} & {} & {} & {} & {} \\\\ \n".format(name, s0old_iter, s0new_iter, s2old_iter, s2new_iter))

# Define which draw picture name and corresponded function
gDict = {"[Graph-linear-vs-log]":draw_linear_vs_log,
         "[Table-MSE-Comparison]": mse_table, 
         "[Table-iter-Comparison]": iter_table,
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
