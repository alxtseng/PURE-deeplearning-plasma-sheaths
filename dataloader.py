#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 18:02:34 2019

@author: alextseng
"""

import numpy as np
import os
import pandas

def load_data(BO_list=[0.01, 0.1, 0.3, 2, 2.5], TE_list=[3, 3, 10, 3, 10], TI_list=[0.3, 1, 0.3, 3, 0.3]):
	##
	# returns:
	#         parameter_list (N x 4): list of parameters ordered as [BO, TE, TI, deg]
	#         iead_list      (N x (500*180)): iead data per set of parameters
	##
	if (len(BO_list) != len(TE_list) or len(TE_list) != len(TI_list) or len(TI_list) != len(BO_list)):
		print("load_data :: ERROR :: receiving parameter lists of different length.")
		return None;

	parameter_list = []
	iead_list = []
	counter = 0

	for idx in range(len(BO_list)):
		BO = BO_list[idx]
		TE = TE_list[idx]
		TI = TI_list[idx]

		parameter_dir = 'IEAD_data/data/[' + str(BO) + ',' + str(TE) + ',' + str(TI) + ']'
		for deg in range(0, 70, 10):
			parameter_list.append([BO, TE, TI, deg])
			deg_str = str(deg)
			if (deg == 0): # fix the 00 problem
				deg_str = '0' + str(deg) 
			deg_dir = 'deg' + deg_str	
			iead = np.genfromtxt(parameter_dir+'/'+deg_dir+'/expid_'+ str(counter) + deg_str+'_IEAD_sp0.dat').reshape((500*180))
			iead = iead / np.max(iead) # normalize
			iead_list.append(iead)
		for deg in range(75, 90, 5):
			parameter_list.append([BO, TE, TI, deg])
			deg_str = str(deg)
			if (deg == 0): # fix the 00 problem
				deg_str = '0' + str(deg) 
			deg_dir = 'deg' + deg_str	
			iead = np.genfromtxt(parameter_dir+'/'+deg_dir+'/expid_'+ str(counter) + deg_str+'_IEAD_sp0.dat').reshape((500*180))
			iead = iead / np.max(iead) # normalize
			iead_list.append(iead)
		counter += 1

	# numpyfy lists
	parameter_list = np.array(parameter_list)
	iead_list = np.array(iead_list)

	for i in range(4):
		parameter_list[:, i] /= np.max(parameter_list[:,i])

	return parameter_list, iead_list