import numpy as np
import pandas as pd
import os


data = pd.read_excel("时间序列用电量数据-小时级别.xlsx")  # 1 3 7 是 预测列
data = data.fillna(-1)
print(data.columns) # Index(['数据采集时间', '每小时的用电量'], dtype='object')
print(data.values)


