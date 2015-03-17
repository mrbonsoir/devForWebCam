'''
Created on 11.01.2015
This module contains function to load data file. 
'''

import sys
import numpy as np
import scipy
import matplotlib.pyplot as plt

def fun_read_input_CS2000(input_file):
	'''
	The input has 2 lines minimum under the following form:
		RGB X Y Z
		<R_value> <G_value> <B_value> <X_value> <Y_value> <Z_value>

	Input:
		data.txt

	Output:
		each line is cast as an element of a list: res_as_list
	'''

	res_as_list = []
	
	f = open(input_file, 'r')
	for line in f:
		a = line.split()
		if a[0] == 'RGB':
			# First line
			print a
		else:
			res_as_list.append([float(a[0]),float(a[1]),float(a[2]),float(a[3]),float(a[4]),float(a[5])])
			print append
		else:
			break
	f.close()

	return res_as_list
