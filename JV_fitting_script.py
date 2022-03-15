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

#define linear equation for fitting resistances
y = lambda x, m, b : m*x + b

#fitting the first column
my_fit, params = opt.curve_fit(J, JV.index.to_list(), JV['1'].values)
fit1 = pd.DataFrame({'1':J(JV.index, my_fit[0], my_fit[1], my_fit[2])}, 
                    index=JV.index.to_list())


JSC = my_fit[0]
VOC = 1/my_fit[1]*np.log(JSC/my_fit[2]+1)


#fitting for Rs
diff = np.abs(JV.index.to_list()-VOC)
Vi = diff.argmin()-3
Vf = diff.argmin()+3
lin_fit, params = opt.curve_fit(y, 
                               JV.index[Vi:Vf+1].to_list(),
                               JV['1'][JV.index[Vi:Vf+1].to_list()].values)

RS = 1/lin_fit[0]*10**3
RS_DF = pd.DataFrame({'Rs_fit':y(JV.index[Vi:Vf+1],lin_fit[0],lin_fit[1])},
                     index=JV.index[Vi:Vf+1].to_list())


#fitting for Rsh
diff = np.abs(JV.index.to_list())
Vi = diff.argmin()-3
Vf = diff.argmin()+3
lin_fit, params = opt.curve_fit(y, 
                               JV.index[Vi:Vf+1].to_list(),
                               JV['1'][JV.index[Vi:Vf+1].to_list()].values)

RSH = 1/lin_fit[0]*10**3
RSH_DF = pd.DataFrame({'Rsh_fit':y(JV.index[Vi:Vf+1],lin_fit[0],lin_fit[1])},
                     index=JV.index[Vi:Vf+1].to_list())


#plotting all fits to see how it looks
plt.figure(figsize=(8,6)).set_facecolor('white')
plt.plot(JV['1'], 'o', label='raw data')
plt.plot(fit1['1'], color='darkred', label='J-V fit')
plt.plot(RS_DF, color='orange', label='Rs fit')
plt.plot(RSH_DF, color='violet', label='Rsh fit')
plt.xlabel('Voltage (V)', fontsize=15)
plt.xticks(fontsize=12)
plt.ylabel('Current Density (mA/cm2)', fontsize=15)
plt.yticks(fontsize=12)
plt.grid(axis='both')
plt.legend(fontsize=12)
plt.savefig('JVfit.png')
plt.show()


#set up characteristics dataframe
chars_index = ['PCE (%)','Jsc (mA/cm2)','Voc (V)','FF (%)','Rsh (Ohm cm2)','Rs (Ohm cm2)']
chars = pd.DataFrame({'1':[0 for i in chars_index], '2':[0 for i in chars_index],
                     '3':[0 for i in chars_index]}, 
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
    
    diff = np.abs(JV.index.to_list())
    Vi = diff.argmin()-3
    Vf = diff.argmin()+3
    lin_fit, params = opt.curve_fit(y, 
                                   JV.index[Vi:Vf+1].to_list(),
                                   JV[col][JV.index[Vi:Vf+1].to_list()].values)
    
    RSH = 1/lin_fit[0]*10**3
    
    diff = np.abs(JV.index.to_list()-VOC)
    Vi = diff.argmin()-3
    Vf = diff.argmin()+3
    lin_fit, params = opt.curve_fit(y, 
                                   JV.index[Vi:Vf+1].to_list(),
                                   JV[col][JV.index[Vi:Vf+1].to_list()].values)
    
    RS = 1/lin_fit[0]*10**3
    
    #write the results to the appropriate column in the characteristics frame
    chars[col] = [PCE,JSC,VOC,FF,RSH,RS]
    
dfi.export(chars,'chars.png')


#taking a ratio of the fitted values to the expected values
compare_DF = data[['1','2','3']][1:8].drop(6)
char_ratio = pd.DataFrame(chars.values/compare_DF.values, 
                         index=['PCE','Jsc','Voc','FF','Rsh','Rs'],
                         columns=['Pixel 1', 'Pixel 2', 'Piexl 3'])
dfi.export(char_ratio,'char_ratio.png')


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#different fitting process - using MPP and only linear fits

power = JV['1']*JV.index
Vi_l1 = JV.index[JV.index < 0][-2]
Vf_l1 = JV.index[JV.index < JV.index[power.argmin()]][-4]
lin1_fit, params = opt.curve_fit(y,
                        JV[Vi_l1:Vf_l1].index,
                        JV['1'][Vi_l1:Vf_l1].values)

JSC = lin1_fit[1]
RSH = 1/lin1_fit[0]*10**3

line1 = pd.DataFrame({'line1':y(JV[Vi_l1:Vf_l1].index, lin1_fit[0], lin1_fit[1])}, index=JV[Vi_l1:Vf_l1].index)

Vi_l2 = JV.index[JV.index > JV.index[power.argmin()]][4]
Vf_l2 = JV['1'][JV['1']>0].index[2]
lin2_fit, params = opt.curve_fit(y,
                        JV[Vi_l2:Vf_l2].index,
                        JV['1'][Vi_l2:Vf_l2].values)

VOC = -lin2_fit[1]/lin2_fit[0]
RS = 1/lin2_fit[0]*10**3

line2 = pd.DataFrame({'line2':y(JV[Vi_l2:Vf_l2].index, lin2_fit[0], lin2_fit[1])}, index=JV[Vi_l2:Vf_l2].index)

VMPP = JV.index[power.argmin()]
JMPP = JV['1'][VMPP]

FF = VMPP*JMPP/(VOC*JSC)*100
PCE = FF*VOC*JSC/100

plt.figure(figsize=(8,6)).set_facecolor('white')
plt.plot(JV['1'], 'o', label='raw data')
plt.plot(VMPP, JMPP, '*', color='violet', ms=12, label='MPP')
plt.plot(line1, color='orange', label='line1')
plt.plot(line2, color='red', label='line2')
plt.xlabel('Voltage (V)', fontsize=15)
plt.xticks(fontsize=12)
plt.ylabel('Current Density (mA/cm2)', fontsize=15)
plt.yticks(fontsize=12)
plt.grid(axis='both')
plt.legend(fontsize=12)
plt.savefig('JVfit_linear.png')
plt.show()

chars_2 = pd.DataFrame({'1':[0 for i in chars_index], '2':[0 for i in chars_index], '3':[0 for i in chars_index]}, 
                     index=chars_index)

for col in JV.columns.to_list():
    power = JV[col]*JV.index
    
    Vi_l1 = JV.index[JV.index < 0][-2]
    Vf_l1 = JV.index[JV.index < JV.index[power.argmin()]][-4]
    lin1_fit, params = opt.curve_fit(y,
                            JV[Vi_l1:Vf_l1].index,
                            JV[col][Vi_l1:Vf_l1].values)
    JSC = -lin1_fit[1]
    RSH = 1/lin1_fit[0]*10**3
    
    Vi_l2 = JV.index[JV.index > JV.index[power.argmin()]][4]
    Vf_l2 = JV[col][JV[col]>0].index[2]
    lin2_fit, params = opt.curve_fit(y,
                            JV[Vi_l2:Vf_l2].index,
                            JV[col][Vi_l2:Vf_l2].values)
    VOC = -lin2_fit[1]/lin2_fit[0]
    RS = 1/lin2_fit[0]*10**3
    
    VMPP = JV.index[power.argmin()]
    JMPP = -JV[col][VMPP]
    
    FF = VMPP*JMPP/(VOC*JSC)*100
    PCE = FF*VOC*JSC/100
    
    chars_2[col] = [PCE,JSC,VOC,FF,RSH,RS]
    
dfi.export(chars_2, 'chars_linear.png')

char_ratio_2 = pd.DataFrame(chars_2.values/compare_DF.values, 
                         index=['PCE','Jsc','Voc','FF','Rsh','Rs'],
                         columns=['Pixel 1', 'Pixel 2', 'Piexl 3'])

dfi.export(char_ratio_2, 'char_ratio_linear.png')