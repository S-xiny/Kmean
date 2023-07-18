#!/usr/bin/env python
# coding: utf-8

# In[84]:


#导包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#读数据，数据文件放在一起n
df = pd.read_excel('data.xlsx')
x = df['s'].values.reshape(-1, 1)
y = df['t'].values.reshape(-1, 1)
#先看下图
plt.figure(figsize=(10,8))
plt.scatter(df['s'],df['t'],color = 'r', alpha=0.5)
#算法
k_means = KMeans(n_clusters=3, random_state=0)
k_means.fit(x,y)
#获取XY数据
z_means = k_means.predict(x)
def find_xy(df):
    data_x = df['s'].values.reshape(-1, 1)
    data_y = df['t'].values.reshape(-1, 1)
    return data_x, data_y
#数据添加到df中
df['label'] = pd.Series(z_means)
data1_x, data1_y = find_xy(df[df['label'] == 0])
data2_x, data2_y = find_xy(df[df['label'] == 1])
data3_x, data3_y = find_xy(df[df['label'] == 2])
#大小
plt.figure(figsize=(10, 8))
#散点图
clt1 = plt.scatter(data1_x, data1_y ,color = 'b', alpha = 0.5)
clt2 = plt.scatter(data2_x, data2_y, color = 'r', alpha = 0.5)
clt3 = plt.scatter(data3_x, data3_y, color = 'm', alpha = 0.5)
#标签
plt.xlabel("s", fontdict={'size': 16})
plt.ylabel("t", fontdict={'size': 16})
#图例
plt.legend(handles=[clt1,clt2, clt3],labels=['well','hard','easy'],loc = 'lower right')
plt.show()
df.to_excel('labeled_data.xlsx')

