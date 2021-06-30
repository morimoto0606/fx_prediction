#%%
# Prepare all Economic data
import pandas as pd
from pylab import mpl, plt
import pandas_datareader.data as web
plt.style.use('seaborn')
%matplotlib inline

start='2009/1/1'
end = '2021/06/20'

df = pd.DataFrame({
    'fx_usdjpy': web.DataReader('DEXJPUS', 'fred', start, end)['DEXJPUS'],
    'fx_eurusd': web.DataReader('DEXUSEU', 'fred', start, end)['DEXUSEU'],
    'fx_gbpusd': web.DataReader('DEXUSUK', 'fred', start, end)['DEXUSUK'],
    'fx_audusd': web.DataReader('DEXUSAL', 'fred', start, end)['DEXUSAL'],
    'fx_nzdusd': web.DataReader('DEXUSNZ', 'fred', start, end)['DEXUSNZ'],
    'ir_jpy': web.DataReader('JPY6MTD156N','fred', start,end)['JPY6MTD156N'],
    'ir_usd': web.DataReader('DGS10','fred',start, end)['DGS10'],
    'oil': web.DataReader('DCOILBRENTEU', 'fred', start,end)['DCOILBRENTEU'],
    'stock_us':web.DataReader("NASDAQ100", 'fred', start,end)['NASDAQ100'],
    'stock_jp':web.DataReader("NIKKEI225","fred",start,end)['NIKKEI225']
    })
df.dropna(inplace=True)

df['fx_usdjpy'].plot()

#%%
# Data preprocessing
import numpy as np
from sklearn.model_selection import train_test_split

# 価格データを対数変化データに変換
df['return_usdjpy'] = np.log(df['fx_usdjpy']).diff()
df['return_eurusd'] = np.log(df['fx_eurusd']).diff()
df['return_gbpusd'] = np.log(df['fx_gbpusd']).diff()
df['return_audusd'] = np.log(df['fx_audusd']).diff()
df['return_nzdusd'] = np.log(df['fx_nzdusd']).diff()
df['return_oil'] = np.log(df['oil']).diff()
df['return_stock_us'] = np.log(df['stock_us']).diff()
df['return_stock_jp'] = np.log(df['stock_jp']).diff()
df['return_ir_jpy'] = df['ir_jpy'].diff()
df['return_ir_usd'] = df['ir_usd'].diff()

# Lag付きのデータの用意(過去3日間分のシフトされたデータ)
lags = 3
for name in df.columns:
    for lag in range(1, lags + 1):
        df['{}_lag{}'.format(name, lag)] = df[name].shift(lag)
df.dropna(inplace=True)

# dataをTraining dataとTest dataに分離しておく
df_train, df_test = train_test_split(df, test_size=0.2, shuffle=False)

from sklearn.preprocessing import StandardScaler
scalar=StandardScaler()
#Training用データから統計量を取得
scalar.fit(df_train) 

# Trainig dataを自分の統計量を使って、平均0, 標準偏差1になるようにスケーリング
df_train_std = pd.DataFrame(scalar.transform(df_train), columns=df.columns)
# Test data はTrading dataと同じパラメータでスケーリング
df_test_std = pd.DataFrame(scalar.transform(df_test), columns=df.columns)

df_train_std.head()


#%%
#correlation 金利差、USDJPY
df_train_std['ir_usd_jpy'] =df_train_std['return_ir_usd'] - df_train_std['return_ir_jpy'] 
df_train_std.plot.scatter(x='ir_usd_jpy', y='return_usdjpy')


#%%
# correlation AUD & NZD
df_train_std.plot.scatter(x='return_audusd', y='return_nzdusd')

#%%
# correlation AUD & OIL
df_train_std.plot.scatter(x='return_oil', y='return_audusd')

#%%
sma_short = 25
sma_long = 250

data = pd.DataFrame({
    'fx_usdjpy': df['fx_usdjpy'],
    'SMA_short': df['fx_usdjpy'].rolling(sma_short).mean(),
    'SMA_long': df['fx_usdjpy'].rolling(sma_long).mean()
})
data.plot()


#%%
#histgram of log return
from scipy.stats import norm

df_train_std['return_usdjpy'].plot.hist(bins=100)
x=np.linspace(-5,5,1000)
plt.plot(x, norm.pdf(x)*400)
print(df_train_std['return_usdjpy'].count())




#%%
# regression(1)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

# 特徴量
attribute = ['return_usdjpy_lag1', 'return_usdjpy_lag2', 'return_usdjpy_lag3']
# Input
X_train = df_train_std[attribute]
X_test = df_test_std[attribute]

# 正解データ
y_train = np.sign(df_train_std['return_usdjpy'])
y_test = np.sign(df_test_std['return_usdjpy'])

# model
regression = LinearRegression()
# Training
regression.fit(X_train, y_train)

# 回帰係数 w
coeff = pd.DataFrame({'x': X_train.columns, 'w': regression.coef_})
coeff.set_index('x').plot.bar()

# Output: 為替予想 up(+1) down(-1)
predict_regression_train = np.sign(regression.predict(X_train))
predict_regression_test = np.sign(regression.predict(X_test))

# Trainig/Test それぞれ正解に対して何割正解したか。
pd.DataFrame({
    'Training': [metrics.accuracy_score(predict_regression_train, y_train)],
    'Test':[metrics.accuracy_score(predict_regression_test, y_test)] }, index={'score'})


#%%
# regression(2)

# 特徴量
attribute = [
'return_usdjpy_lag1', 'return_usdjpy_lag2', 'return_usdjpy_lag3',
'return_eurusd_lag1', 'return_eurusd_lag2', 'return_eurusd_lag3',
'return_gbpusd_lag1','return_gbpusd_lag2','return_gbpusd_lag3',
'return_audusd_lag1','return_audusd_lag2','return_audusd_lag3',
'return_oil_lag1','return_oil_lag2','return_oil_lag3',
'return_ir_jpy_lag1','return_ir_jpy_lag2','return_ir_jpy_lag3',
'return_ir_usd_lag1','return_ir_usd_lag2','return_ir_usd_lag3',
'return_stock_us_lag1','return_stock_us_lag2','return_stock_us_lag3',
'return_stock_jp_lag1','return_stock_jp_lag2', 'return_stock_jp_lag3']

# Input
X_train = df_train_std[attribute]
X_test = df_test_std[attribute]

regression2= LinearRegression() # model
regression2.fit(X_train, y_train) # Training

# 回帰係数 w
coeff = pd.DataFrame({'x': X_train.columns, 'w': regression2.coef_})
coeff.set_index('x').plot.bar()

# Output: 為替予想 up(+1) down(-1)
predict_regression2_train = np.sign(regression2.predict(X_train))
predict_regression2_test = np.sign(regression2.predict(X_test))

pd.DataFrame({
    'Training': [metrics.accuracy_score(predict_regression2_train, y_train)],
    'Test':[metrics.accuracy_score(predict_regression2_test, y_test)] }, index={'score'})


#%% 
# Lasso
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.01) # model
lasso.fit(X_train, y_train) # Training

coeff = pd.DataFrame({'x': X_train.columns, 'w': lasso.coef_}) # 回帰係数
coeff.set_index('x').plot.bar()
 
# Output: 為替予想 up(+1) down(-1)
predict_lasso_train = np.sign(lasso.predict(X_train))
predict_lasso_test = np.sign(lasso.predict(X_test))

pd.DataFrame({
    'Training': [metrics.accuracy_score(predict_lasso_train, y_train)], #Traing dataに対する正解率
    'Test':[metrics.accuracy_score(predict_lasso_test, y_test)] }, # Test dataに対する正解率
    index={'score'})

#%%
# Decision Tree with GridsearchCV
from sklearn import tree
from sklearn.model_selection import GridSearchCV

# 特徴量
attribute = [
    'return_usdjpy_lag1',
    'return_eurusd_lag1',
    'return_gbpusd_lag1',
    'return_audusd_lag1',
    'return_oil_lag1',
    'return_ir_jpy_lag1',
    'return_ir_usd_lag1',
    'return_stock_us_lag1',
    'return_stock_jp_lag1']

# Input
X_train = df_train_std[attribute]
X_test = df_test_std[attribute]

# Hyper parameterの候補
hyper_params = {
    'random_state': [1024], #model内部で乱数を使うときのシード
    'min_samples_split': [3, 5, 10, 15, 20], #nodeを分割するのに必要なパラメータの最小値
    'max_depth': [10, 20, 30] #Treeの深さの最大値
}

# GridSearchCVのConstructor
clf_tree = GridSearchCV(tree.DecisionTreeClassifier(),hyper_params, cv=5, verbose=False,refit=True)       
# 各Hyper parameterの組み合わせでTraing
clf_tree.fit(X_train, y_train) 

# bestなHyperParametrでのTraining結果に対する特徴量の重要度
importance = clf_tree.best_estimator_.feature_importances_
pd.Series(importance, index = X_train.columns).plot.bar()
print(clf_tree.best_params_) #bestなHyperParameterの表示

# Output: 為替予想 up(+1) down(-1)
predict_tree_train = clf_tree.best_estimator_.predict(X_train)
predict_tree_test = clf_tree.best_estimator_.predict(X_test)

pd.DataFrame({
    'Training': [metrics.accuracy_score(predict_tree_train, y_train)], #Traing dataに対する正解率
    'Test':[metrics.accuracy_score(predict_tree_test, y_test)] }, #Test dataに対する正解率
    index={'score'}) 

#%%
# Random forest with GridsearchCV
from sklearn.ensemble import RandomForestClassifier as RFC

# Hyper parameterはDecisionTreeで見つけた
hyper_params = {
    'random_state': [clf_tree.best_params_['random_state']], #内部で使われる乱数のシード
    'min_samples_split': [clf_tree.best_params_['min_samples_split']], #nodeを分割するのに必要なパラメータの最小値
    'max_depth': [clf_tree.best_params_['max_depth']] #Treeの深さの最大値
}

# GridSearchCVのConstructor
clf_rf = GridSearchCV(RFC(), hyper_params, cv=5, verbose=False, refit=True, n_jobs=-1)       
clf_rf.fit(X_train, y_train) # 各Hyper parameterの組み合わせでTraing

# bestなHyperParametrでのTraining結果に対する特徴量の重要度
importance = clf_rf.best_estimator_.feature_importances_
pd.Series(importance, index = X_train.columns).plot.bar()
print(clf_rf.best_params_) #bestなHyperParameterの表示

# Output: 為替予想 up(+1) down(-1)
predict_rf_train = clf_rf.best_estimator_.predict(X_train) 
predict_rf_test = clf_rf.best_estimator_.predict(X_test) 

pd.DataFrame({
    'Training': [metrics.accuracy_score(predict_rf_train, y_train)], #trainig dataに対する予想
    'Test':[metrics.accuracy_score(predict_rf_test, y_test)] }, #test dataに対する予想
    index={'score'})

#%%
# NeuralNetwork with GridsearchCV
from sklearn.neural_network import MLPClassifier

# Hyper parameterの候補
tuned_parameters =     {
    'solver':['sgd'],   # 最適化手法 stochastic gradient desent
    'activation': ['logistic', 'tanh'], # activation function
    # 隠れ層の層の数と、各層のニューロンの数
    'hidden_layer_sizes':[(50, 100, 50), (100, 50, 50),(100, 50, 100)], 
    'random_state':[1024], #内部で使われる乱数のシード
    'max_iter':[10000] # 最適化時の最大イテレーション数
}

# GridSearchCVのConstructor
clf_nn=GridSearchCV(MLPClassifier(), param_grid=tuned_parameters, scoring='accuracy', cv=5, refit=True, n_jobs=-1)
clf_nn.fit(X_train, y_train) # 各Hyper parameterの組み合わせでTraing
print(clf_nn.best_params_) #bestなHyperParameterの表示

# Output: 為替予想 up(+1) down(-1)
predict_nn_train = clf_nn.best_estimator_.predict(X_train)
predict_nn_test = clf_nn.best_estimator_.predict(X_test)

pd.DataFrame({
    'Training': [metrics.accuracy_score(predict_nn_train, y_train)], #trainig dataに対する予想
    'Test':[metrics.accuracy_score(predict_nn_test, y_test)] }, #test dataに対する予想
    index={'score'})


#%%
#Back Test
sma_short = 25
sma_long = 250

data = pd.DataFrame({
    'fx_usdjpy': df['fx_usdjpy'],
    'SMA_short': df['fx_usdjpy'].rolling(sma_short).mean(),
    'SMA_long': df['fx_usdjpy'].rolling(sma_long).mean()
})

data.dropna(inplace=True)
# position 
data['position'] = np.where(data['SMA_short'] > data['SMA_long'], 1, -1)
data.plot(secondary_y='position', figsize=(8,4))

data['base'] = df['fx_usdjpy'].diff()
data['sma'] = data['position'].shift(1) * data['base']

data.dropna(inplace = True)
data[['base', 'sma']].cumsum().plot(figsize=(8, 4))


#%%
# PL
dX0, dX = train_test_split(df['fx_usdjpy'].diff(), test_size=0.2, shuffle=False)
pl = pd.DataFrame({'base': dX})
pl['regression'] = predict_regression_test * dX
pl['regression2'] = predict_regression2_test * dX
pl['lasso'] = predict_lasso_test * dX
pl['tree'] = predict_tree_test * dX
pl['rf'] = predict_rf_test * dX
pl['nn'] = predict_nn_test * dX

np.cumsum(pl).plot(figsize=(9, 6))
