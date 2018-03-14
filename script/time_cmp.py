#!/usr/bin/python3
import sys
import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as tic

out_path = "/home/johncreed/liblr_test/out_p/"
pictures_path = "/home/johncreed/liblr_test/pictures_p/"
file_names = [f for f in listdir(out_path)]
best_result = {}
result_record = {}
name_list = []

def parce_name( file ):
		name = ""
		ls = file.split("_")
		for x in ls[:-1]:
				if(name == ""):
						name = x
				else:
						name = name+"_"+x
		return [name, ls[-1]]

def print_best_result():
		for i in best_result:
				print (i ,best_result[i])

def cmp_fun( ls ):
		return float(ls.split()[0])

def print_result_record():
		for i in best_result:
				print (i, best_result[i])
				for j in result_record:
						if( parce_name(j)[0] == i ):
								print(result_record[j])

def find_best_result():
		for file in file_names:
				with open(out_path+file) as f:
						for line in f:
								tmp = line.split()
								if(len(tmp) == 0): 
										continue
								if(tmp[0] == "Best"):
										name = parce_name(file)
										result = float(tmp[-1])
										key = name[0]
										value = [name[1], result]
										if(key not in best_result.keys()):
												best_result[key] = value
										else:
												if (result < best_result[key][1]):
														best_result[key] = value
										if(file not in result_record.keys()):
												result_record[file] = [file, result]

find_best_result()

print_result_record()

def draw_pic():
		for file_name in file_names:
				print (file_name)
				version = 0
				sum0 = 0
				sum1 = 0

				y0 = []
				y1 = []
				err = []
				x = []
				init_c = 30
				break_c = 30
				best_c = 30
				
				with open(out_path+file_name) as f:
						for line in f:
								if (line[0] == "+"):
										version = (version + 1) % 2
								if ("iter" in line):
										tmp = line.strip().split()
										if (version == 0):
												sum0 += int( tmp[-1] )
										else:
												sum1 += int( tmp[-1] )
								if ("log2c=" in line):
										tmp = line.strip().split()
										if(init_c == 30): init_c = float(tmp[1])
										y0.append(sum0)
										y1.append(sum1)
										MSE = (tmp[-1].split("="))[-1]
										err.append( MSE )
										x.append( tmp[1] )
								if ("break with" in line):
										tmp = line.strip().split()
										break_c = int(tmp[-1])
								if("log2 Best C" in line):
										tmp = line.strip().split()
										best_c = int(tmp[-1])
				
				# set x axis size
				end_c = break_c if break_c >= best_c else best_c
				c_range = int(end_c - init_c + 5)
				#c_range = len(y0)
				y0_arr = np.array(y0)
				y1_arr = np.array(y1)
				err_arr = np.array(err)
				x_arr = np.array(x)

				print ("y0 size %d", y0_arr.size)
				print ("y1 size %d", y1_arr.size)
				print ("x size %d", x_arr.size)
				
				# get the best record of data
				f_n = parce_name(file_name)
				if f_n[0] in best_result.keys():
						best_rc = best_result[f_n[0]]
				else:
						print ("quit")
						continue
				
				# draw picture
				fig, ax = plt.subplots()
				plt.title("["+file_name+", "+str(result_record[file_name][1])+"]"+" "+"["+best_rc[0]+", "+str(best_rc[1])+"]")
				ax.set_xlabel('Log2 p')
				ax.set_ylabel('Cumulative CG Iterator')
				line1 = ax.plot(x_arr, y0_arr, 'b--o',linewidth=2,
								label='Warm Start')
				line2 = ax.plot(x_arr, y1_arr, 'g--o', linewidth=2,
								label='No Warm Start')
				
				formatter = tic.ScalarFormatter()
				formatter.set_scientific(True)
				formatter.set_powerlimits((0,0))
				ax2 = ax.twinx()
				ax2.yaxis.set_major_formatter(formatter)
				ax2.set_ylabel("MSE")
				line3 = ax2.plot(x_arr, err_arr, 'y-x', linewidth=2,
								label='MSE')
				
				break_c_label = "Break Log2_P: " +	str(break_c)
				best_c_label = "Best Log2_P: " +	str(best_c)
				#v1= plt.axvline(x=break_c, linewidth=2, color="r", label=break_c_label)
				v2= plt.axvline(best_c, linewidth=2, color="black", label= best_c_label)
				
				lns = line1+line2+line3
				#lns.append(v1)
				lns.append(v2)
				labs = [l.get_label() for l in lns]
				plt.legend(lns, labs, loc="upper left")

				fig_name = file_name + ".eps"
				plt.savefig(pictures_path+fig_name, format="eps", dpi=1000)
				plt.close();

draw_pic()
