from sklearn import preprocessing
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from datetime import datetime
import time
import math
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from scipy import stats, integrate
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error  # 评价指标
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
from date_process import data_read_csv
train_x,train_y=data_read_csv()
x_train, x_test, y_train, y_test = train_test_split(np.array(train_x), np.array(train_y), test_size=0.4,shuffle=True,random_state=1)
print('x_train.shape',x_train.shape)
print('x_test.shape',x_test.shape)
# 集成学习模型
# svm算法

from sklearn.linear_model import LassoLarsIC as LR#逻辑回归
svm = LGBMRegressor(n_estimators=10)
svm.fit(x_train, y_train)
svm_pred = svm.predict(x_test)

from metra import metric
mae, mse, rmse, mape, mspe,r2=metric(np.array(svm_pred), np.array(y_test))
print('mae, mse, rmse, mape, mspe,r2')
print(mae, mse, rmse, mape, mspe,r2)
# 设置Seaborn样式
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid")
x = range(len(y_test))
data = pd.DataFrame({'x': x, 'y_pred': svm_pred.flatten(), 'y_true': y_test.flatten()}).iloc[100:200,:]
# 绘制y_pred的折线图
sns.lineplot(x='x', y='y_pred', data=data, linewidth=1, label='y_pred')

# 绘制y_true的折线图
sns.lineplot(x='x', y='y_true', data=data, linewidth=1, label='y_true')

# 添加标题和标签
plt.title('Prediction vs True')
plt.xlabel('Date')
plt.ylabel('Values')
plt.savefig('AdaBoost_Prediction_True.png')
# 显示图形
plt.show()