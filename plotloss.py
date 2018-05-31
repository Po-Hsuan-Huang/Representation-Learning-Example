#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 17:25:50 2018

@author: pohsuanh
"""
import matplotlib.pyplot as plt
fullDict={}
files = ['loss_function.txt', 'refine_loss_fun.txt']
for fi in files :
    tempDict={}
    Epoch = []
    TotalLoss = []
    ReconstructionLoss = []
    LatentLoss = []
            
    for line in open(fi,'r'):
        if 'loss' in line:

            dataList=[dataItem.strip() for dataItem in line.strip().split(' ') if dataItem != '']
            Epoch.append( dataList[0] )
            TotalLoss.append( dataList[4] )
            ReconstructionLoss.append( dataList[7])
            LatentLoss.append( dataList[10])
    tempDict={'Epoch':Epoch, 'TotalLoss': TotalLoss, 'ReconstructionLoss': ReconstructionLoss, 'LatentLoss':LatentLoss
             } 

    fullDict[fi.split('.')[0]]=tempDict
            
Loss_ogn = fullDict['loss_function']
Loss_rfn = fullDict['refine_loss_fun']
x =list( range(len(Loss_ogn['Epoch'])))
f,ax = plt.subplots(3, sharey =True)
ax[0].plot(x ,Loss_ogn['TotalLoss'],x,Loss_rfn['TotalLoss'],'-r','-b')
ax[0].set_title('Total Loss')
ax[0].legend(('refine', 'origin'), loc = 'upper right')
ax[1].plot(x ,Loss_ogn['ReconstructionLoss'],x,Loss_rfn['ReconstructionLoss'],'-r','-b')
ax[1].set_title('ReconstructionLoss')
ax[1].legend(('refine', 'origin'), loc = 'upper right')
ax[2].plot(x ,Loss_ogn['LatentLoss'],x,Loss_rfn['LatentLoss'],'-r','-b')
ax[2].set_title('LatentLoss')
ax[2].legend(('refine', 'origin'), loc = 'upper right')
