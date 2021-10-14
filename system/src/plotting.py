# -*- coding: utf-8 -*-
"""
File: plotting.py
Project: Object Recognition System
Date: 12/08/2021
Author: Diego Bueno da Silva
e-mail: d.bueno.da.silva.10@student.scu.edu.au
ID: 23567850

Plotting functions 

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

""" Function plotScatterChatResultsComparisson(   )

    Plot a chart showing Hyperparameters versus its performance
    
    parameters: results - results CSV file
                x - column name of x axis
                y - column name of values of y axis
                title - title of the chart
                xLabel - description of label x
                yLabel - description of label y

    return: 
        none
    
"""
def plotScatterChatResultsComparisson(results,x,y,xLabel, yLabel,mean,legend):
    
    # reading the results and plot the comparison with accuracy
    resultsDf = pd.read_csv(results)
        
    # Define llenDf colours 
    jet = plt.get_cmap('jet')
    colors = iter(jet(np.linspace(0,1,len(resultsDf.index))))
    
    plt.title('')        
    plt.xlabel(xLabel) 
    plt.ylabel(yLabel)
    
    for index, row in resultsDf.iterrows():
       # Plotting values in different colours
       plt.scatter(row[x],row[y], color= next(colors) )
     
    avg = np.mean(resultsDf[y])    
     
    if mean == 1:
          # plot the ideal case in red color
          plt.plot(resultsDf[x], [avg for _ in range(len(resultsDf[x]))], color='red')
    
    plt.figlegend(resultsDf[legend],loc='upper left')    
    
    plt.show()
            

""" Function plotBarChatResultsComparisson(   )

    Plot a bar chart showing categorical Hyperparameters versus its performance
    
    parameters: results - results CSV file
                x_names - array with column of category names of x axis
                y - column name of values of y axis
                title - title of the chart
                xLabel - description of label x
                yLabel - description of label y

    return: 
        none
    
"""
def plotBarChatResultsComparisson(title,results,x,y,xLabel,err):

    # reading the results and plot the comparison with accuracy
    resultsDf = pd.read_csv(results) 
       
    plt.rcdefaults()
    fig, ax = plt.subplots()
    
    y_pos = np.arange(len(resultsDf[y]))
    
    ax.barh(resultsDf[y], resultsDf[x]*100, xerr=resultsDf[err], align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(resultsDf[y])
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel(xLabel)
    ax.set_title(title)
    
    plt.show()    
   
""" Function plotPieChatResultsComparisson(   )

    Plot a pie chart showing 
    
    parameters: results - results CSV file
                x_names - array with column of category names of x axis
                y - column name of values of y axis

    return: 
        none
    
"""
def plotPieChatResultsComparisson(results,x,y):
    
    # reading the results and plot the comparison with accuracy
    resultsDf = pd.read_csv(results)
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.axis('equal')
    ax.pie(resultsDf[y], labels = resultsDf[x]) #,autopct='%1.2f%%'
    plt.show()
    

# Plotting Pie Chart
pathResults = '/Users/diego/workspace/spyder/COMP6006-ExpertAndIntelligentSysApp/system/results/'
fileName = '2021-10-14_1928.csv' 
results = pathResults + fileName 
x = 'Image Feature and Preprocessing approch'
y = 'Epochs'  
plotPieChatResultsComparisson(results,x,y)    
    
    
pathResults = '/Users/diego/workspace/spyder/COMP6006-ExpertAndIntelligentSysApp/system/results/'
fileName = '2021-10-14_1928.csv' 
results = pathResults + fileName 
x = 'Epochs'
y = 'Accuracy'
legend = 'Image Feature and Preprocessing approch'
xLabel = 'Number of Epochs executed'
yLabel = 'Accuracy'
mean = 0

# Plotting Scatter
plotScatterChatResultsComparisson(results,x,y,xLabel,yLabel,mean,legend)

y = 'Image Feature and Preprocessing approch'
x = 'Accuracy'
xLabel = 'Accuracy Achieved'
title = 'Accuracy by Image Feature'
err = 'Epochs'

# Plotting bar chart
plotBarChatResultsComparisson(title,results,x,y,xLabel,err)
        
    
    
    
    
    