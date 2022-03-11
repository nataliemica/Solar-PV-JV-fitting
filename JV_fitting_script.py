# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 15:22:38 2022

@author: Natalie
"""

import pandas as pd
import pathlib
import dataframe_image as dfi

import numpy as np

import scipy.optimize as opt

import matplotlib.pyplot as plt

#find the data we want to fit, organize the dataframe
data = pd.read_table(str(pathlib.Path().resolve())+'/data/example.txt', delimiter='\t', lineterminator='\n')
header = ['Pixel','1','2','3','Pixel.1','1.1','2.1','3.1']
data.columns = header

#select the portion of the above dataframe that is the JV curves
JV = data[['Pixel','1','2','3']][10:]
JV.index = np.around(np.arange(float(JV.Pixel[10]),
                     float(JV.Pixel[len(JV)+9]),
                     float(JV.Pixel[11])-float(JV.Pixel[10])), decimals=3)
JV = JV.drop('Pixel', axis=1)

#define ideal diode equation
J = lambda V, Jsc, A, Jd : Jd*(np.exp(A*V)-1)-Jsc

#fitting the first column
my_fit, params = opt.curve_fit(J, JV.index.to_list(), JV['1'].values)
fit1 = pd.DataFrame({'1':J(JV.index, my_fit[0], my_fit[1], my_fit[2])}, index=JV.index.to_list())

plt.figure(figsize=(8,6)).set_facecolor('white')
plt.plot(JV['1'], 'o', label='raw data')
plt.plot(fit1['1'], color='darkred', label='fit')
plt.xlabel('Voltage (V)', fontsize=15)
plt.xticks(fontsize=12)
plt.ylabel('Current Density (mA/cm2)', fontsize=15)
plt.yticks(fontsize=12)
plt.grid(axis='both')
plt.legend(fontsize=12)
plt.savefig('JVfit.png')
plt.show()

#set up characteristics dataframe
chars_index = ['PCE (%)','Jsc (mA/cm2)','Voc (V)','FF (%)']
chars = pd.DataFrame({'1':[0 for i in chars_index], '2':[0 for i in chars_index], '3':[0 for i in chars_index]}, 
                     index=chars_index)

for col in JV.columns.to_list():
    #fit the JV data in one column
    my_fit, params = opt.curve_fit(J, JV.index.to_list(), JV[col].values)
    JSC = my_fit[0]
    VOC = 1/my_fit[1]*np.log(JSC/my_fit[2]+1)    
    
    #find the power generated from the fitted equation
    power_x = np.arange(0,VOC,0.01)
    power = J(power_x, my_fit[0], my_fit[1], my_fit[2])*power_x
    
    VMPP = power_x[power.argmin()]
    JMPP = np.abs(J(VMPP, my_fit[0], my_fit[1], my_fit[2]))
    
    FF = (VMPP*JMPP)/(VOC*JSC)*100
    PCE = FF*JSC*VOC/100
    
    #write the results to the appropriate column in the characteristics frame
    chars[col] = [PCE,JSC,VOC,FF]
    
dfi.export(chars,'chars.png')

#taking a ratio of the fitted values to the expected values
char_ratio = pd.DataFrame(chars.values/data[['1','2','3']].head()[1:5].values, 
                         index=['PCE','Jsc','Voc','FF'],
                         columns=['Pixel 1', 'Pixel 2', 'Piexl 3'])
dfi.export(char_ratio,'char_ratio.png')
char_ratio