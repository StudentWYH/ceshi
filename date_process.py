import numpy as np
import pandas as pd
import os
# data = pd.read_excel("时间序列用电量数据-小时级别.xlsx")  # 1 3 7 是 预测列
# data = data.fillna(-1)
# prin_t(data.columns) # Index(['数据采集时间', '每小时的用电量'], dtype='object')
# print(data.values)


def data_read_csv():
       data = pd.read_excel("河南空气质量(2023-06-01)(1).xlsx")  # 1 3 7 是 预测列
       data = data.fillna(0)
       print(data.columns)  # Index(['数据采集时间', '每小时的用电量'], dtype='object')
       print(data.values)
       datax=data[['pm10', 'so2', 'no2', 'co', 'o3']].values[:2000]
       datay=data[['pm2_5']].values[:2000]
       return datax,datay
data_read_csv()
