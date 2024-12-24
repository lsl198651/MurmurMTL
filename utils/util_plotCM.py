#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   util_plotCM.py
@Contact :   tpian233@163.com

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2023/1/4 19:55   tian      1.0         None
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

inputBase = '..\..\logs\pyoutput\FuseV4BiGRU2D_2000Hz10sV2_EnvolopeMelSpecg80msPCA_202212271951'
cmname = 'GTPredict_150'

csvPath = os.path.join(inputBase,cmname+'.csv')
confusion = pd.read_csv(csvPath)
confusionData = confusion.values[1:6,2:6].T.astype(int)
plt.imshow(confusionData, cmap=plt.cm.Blues)
# ticks 坐标轴的坐标点
# label 坐标轴标签说明
indices = range(len(confusionData))
# 第一个是迭代对象，表示坐标的显示顺序，第二个参数是坐标轴显示列表
plt.xticks(indices, ['S1', 'Systole', 'S2', 'Diastole'])
plt.yticks(indices, ['S1', 'Systole', 'S2', 'Diastole'])

plt.xlabel('Predict State')
plt.ylabel('Ground Truth')
# plt.title('Comfusion Mattrix')

# plt.rcParams两行是用于解决标签不能显示汉字的问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 800
# 显示数据
thresh = confusionData.max() / 2.
for first_index in range(len(confusionData)):    #第几行
    for second_index in range(len(confusionData[first_index])):    #第几列
        plt.text(first_index,
                 second_index,
                 confusionData[first_index][second_index],
                 horizontalalignment="center",
                 color="white" if confusionData[first_index][second_index] > thresh else "black")
# 在matlab里面可以对矩阵直接imagesc(confusion)
# 显示
plt.savefig(os.path.join(inputBase,'figure',cmname+'.png'))
plt.show()
plt.close()