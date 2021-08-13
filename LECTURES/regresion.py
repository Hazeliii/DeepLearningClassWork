import math
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import datetime

# 修改matplotlib样式
style.use('ggplot')
# df = quandl.get('WIKI/GOOGL')
# df.to_csv("googlestock.csv", index=False, header=False)
# df.to_csv("googlestock2.csv")
df = pd.read_csv("googlestock2.csv")
df['Date'] = pd.to_datetime(df['Date'])
df.set_index("Date", inplace=True)

# 定义预测列变量，它存放研究对象的标签名
forecast_col = 'Adj. Close'
# 定义预测天数，这里设置为所有数据量长度的10%
forecast_out = int(math.ceil(0.1*len(df)))

# 只用到df中下面的几个字段
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

# 构造两个新的列
# HL_PCT为股票最高价与最低价的变化百分比
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
# HL_PCT为股票收盘价与开盘价的变化百分比
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
# 下面为真正用到的特征字段
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
# 因为scikit-learn并不会处理空数据，需要把为空的数据都设置为一个比较难出现的值，这里取-9999，
df.fillna(-99999, inplace=True)
# 用label代表该字段，是预测结果
# 通过让与Adj. Close列的数据往前移动1%行来表示
df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'], 1))
#按列去scale表，减均值，除方差
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
# 抛弃label列中为空的那些行
#df.dropna(inplace=True)
y = np.array(df['label'])
y = y[:-forecast_out]

X_train, X_val, y_train, y_val = model_selection.train_test_split(X, y, test_size=0.2)

clf = LinearRegression(n_jobs=-1) # LinearRegression 97.7%
#clf = svm.SVR() # Support vector machine 79.5%
#clf = svm.SVR(kernel='poly') # Support vector machine 68.5%
clf.fit(X_train, y_train) # Here, start training
accuracy = clf.score(X_val, y_val) # Test an get a score for the classfier
forecast_set = clf.predict(X_lately)

print(forecast_set, accuracy)


# 在df中新建Forecast列，用于存放预测结果的数据
df['Forecast'] = np.nan
df['Forecast'][-forecast_out:] = forecast_set


# 开始绘图
# df.to_csv("googlestock_forecast.csv", index=False, header=False)
df.to_csv("googlestock_forecast_tf.csv")
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
