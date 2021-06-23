#%%
from operator import mod, pos
from re import I
import numpy as np
from numpy import random
import pandas as pd
from pylab import mpl, plt
import pandas_datareader.data as web
plt.style.use('seaborn')
%matplotlib inline

# Data のロード　ここではFredというサイトから、日次のマーケットデータ
start='2009/1/1'
end='2020/6/20'
df = pd.DataFrame({
    'fx_usdjpy': web.DataReader('DEXJPUS', 'fred', start, end)['DEXJPUS'],
    'fx_eurusd': web.DataReader('DEXUSEU', 'fred', start, end)['DEXUSEU'],
    'fx_gbpusd': web.DataReader('DEXUSUK', 'fred', start, end)['DEXUSUK'],
    'fx_audusd': web.DataReader('DEXUSAL', 'fred', start, end)['DEXUSAL'],
    'fx_nzdusd': web.DataReader('DEXUSNZ', 'fred', start, end)['DEXUSNZ'],
    'ir_jpy': web.DataReader('JPY6MTD156N','fred', start, end)['JPY6MTD156N'],
    'ir_usd': web.DataReader('DGS10','fred', start, end)['DGS10'],
    'oil': web.DataReader('DCOILBRENTEU', 'fred', start, end)['DCOILBRENTEU'],
    'stock_us':web.DataReader("NASDAQ100", 'fred', start, end)['NASDAQ100'],
    'stock_jp':web.DataReader("NIKKEI225","fred",start, end)['NIKKEI225']
    })
df.dropna(inplace=True)


#%%

df['return_usdjpy'] = np.log(df['fx_usdjpy']).diff()
df['return_eurusd'] = np.log(df['fx_eurusd']).diff()
df['return_gbpusd'] = np.log(df['fx_gbpusd']).diff()
df['return_audusd'] = np.log(df['fx_audusd']).diff()
df['return_nzdusd'] = np.log(df['fx_nzdusd']).diff()
df['return_ir_jpy'] = df['ir_jpy'].diff()
df['return_ir_usd'] = df['ir_usd'].diff()
df['return_oil'] = np.log(df['oil']).diff()
df['return_stock_us'] = np.log(df['stock_us']).diff()
df['return_stock_jp'] = np.log(df['stock_jp']).diff()
df['return_ir_jpy'] = df['ir_jpy'].diff()
df['return_ir_usd'] = df['ir_usd'].diff()
df.dropna(inplace=True)

df.head()
df.tail()

#%%
df.describe()
#%%
# Lag付きのデータの用意
lags = 5 
cols = []
names = df.columns
for n in names:
    for lag in range(1, lags + 1):
        name = '{}_lag{}'.format(n, lag)
        df[name] = df[n].shift(lag)
        cols.append(name)
df.dropna(inplace=True)
X  = df[cols]
y = df['return_usdjpy']
df.head()

#%%
print(cols)


#%%
# standartd scalr, Scale change
from sklearn.preprocessing import StandardScaler
scalar=StandardScaler()
scalar.fit(df)
scalar.mean_
scalar.transform(df)
df_mean = pd.DataFrame(scalar.mean_.reshape(1, len(df.columns)), columns=df.columns)
df_scale= pd.DataFrame(scalar.scale_.reshape(1, len(df.columns)), columns=df.columns) 
df_std = pd.DataFrame(scalar.transform(df), columns=df.columns)
df_std.head()


#%%
# Regression to return
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics


model = LinearRegression()
print(df_std.columns)
X = df_std[['return_usdjpy_lag1', 'return_usdjpy_lag2', 'return_usdjpy_lag3']]
y = df_std['return_usdjpy']
x_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


model.fit(x_train, y_train)
predict = model.predict(X_test)
print('train score', model.score(x_train, y_train))
print('test score', model.score(X_test, y_test))

print(metrics.classification_report(np.sign(y_test), np.sign(predict)))


#%%
# Regression to direction
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics


model = LinearRegression()
print(df_std.columns)
X = df_std[['return_usdjpy_lag1', 'return_usdjpy_lag2', 'return_usdjpy_lag3']]
y = np.sign(df_std['return_usdjpy'])
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, shuffle=False)


model.fit(X_train, y_train)
predict = model.predict(X_test)
print('score InSample', model.score(X_train, y_train))
print('score OutOfSample', model.score(X_test, y_test))

print(metrics.classification_report(np.sign(y_test), np.sign(predict)))

#%%
# Regression to return
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics


regression = LinearRegression()
X = df_std[[
    'return_usdjpy_lag1', 'return_usdjpy_lag2', 'return_usdjpy_lag3',
    'return_eurusd_lag1', 'return_eurusd_lag2', 'return_eurusd_lag3',
    'return_gbpusd_lag1','return_gbpusd_lag2','return_gbpusd_lag3',
    'return_ir_jpy_lag1','return_ir_jpy_lag2','return_ir_jpy_lag3',
    'return_ir_usd_lag1','return_ir_usd_lag2','return_ir_usd_lag3',
    'return_stock_us_lag1','return_stock_us_lag2','return_stock_us_lag3',
    'return_stock_jp_lag1','return_stock_jp_lag2', 'return_stock_jp_lag3']]

y = df['return_usdjpy']
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, shuffle=False)


regression.fit(X_train, y_train)
print('score InSample', metrics.accuracy_score(np.sign(regression.predict(X_train)), np.sign(y_train)))
print('score OutOfSample', metrics.accuracy_score(np.sign(regression.predict(X_test)), np.sign(y_test)))

coeff = pd.DataFrame({'x': X.columns, 'y': regression.coef_})
coeff.set_index('x').plot.bar()


position = np.sign(regression.predict(X_test))
v1, v2, v3, dX = train_test_split(X, df['fx_usdjpy'].diff(), test_size=0.2, shuffle=False)

strategy = position * dX
result = pd.DataFrame({'strategy': strategy,  'return': dX})
np.cumsum(result).plot()






#%%
# Regression to direction
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

print('Lenar regress')
regression= LinearRegression()
print(df_std.columns)
X = df_std[[
    'return_usdjpy_lag1', 'return_usdjpy_lag2', 'return_usdjpy_lag3',
    'return_eurusd_lag1', 'return_eurusd_lag2', 'return_eurusd_lag3',
    'return_gbpusd_lag1','return_gbpusd_lag2','return_gbpusd_lag3',
    'return_audusd_lag1','return_audusd_lag2','return_audusd_lag3',
    'return_oil_lag1','return_oil_lag2','return_oil_lag3',
    'return_ir_jpy_lag1','return_ir_jpy_lag2','return_ir_jpy_lag3',
    'return_ir_usd_lag1','return_ir_usd_lag2','return_ir_usd_lag3',
    'return_stock_us_lag1','return_stock_us_lag2','return_stock_us_lag3',
    'return_stock_jp_lag1','return_stock_jp_lag2', 'return_stock_jp_lag3']]


y = np.sign(df['return_usdjpy'])
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, shuffle=False)


regression.fit(X_train, y_train)
print('score InSample', metrics.accuracy_score(np.sign(regression.predict(X_train)), np.sign(y_train)))
print('score OutOfSample', metrics.accuracy_score(np.sign(regression.predict(X_test)), np.sign(y_test)))

coeff = pd.DataFrame({'x': X.columns, 'y': regression.coef_})
coeff.set_index('x').plot.bar()


position = np.sign(regression.predict(X_test))
v1, v2, v3, dX = train_test_split(X, df['fx_usdjpy'].diff(), test_size=0.2, shuffle=False)

strategy = position * dX
result = pd.DataFrame({'strategy': strategy,  'return': dX})
np.cumsum(result).plot()



#%%
# Regression to direction
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics


model = LogisticRegression()
X = df_std[[
    'return_usdjpy_lag1', 'return_usdjpy_lag2', 'return_usdjpy_lag3']]
#X = df_std[[
#    'return_usdjpy_lag1',
#    'return_eurusd_lag1',
#    'return_gbpusd_lag1']]

y = np.sign(df['return_usdjpy'])
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, shuffle=False)



model.fit(X_train, y_train)
print('score InSample', model.score(X_train, y_train))
print('score OutOfSample', model.score(X_test, y_test))
print('coeff', model.coef_)
predict = (model.predict_proba(X_test))[0:]

position = predict
v1, v2, v3, dX = train_test_split(X, df['fx_usdjpy'].diff(), test_size=0.2, shuffle=False)
print(dX)
print(predict)
pl1 = predict[:, 0] * dX
result = pd.DataFrame({'strategy': pl1,  'return': dX})
np.cumsum(result).plot()



#%%
# lasso to scaled data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
X = df_std[[
    'return_usdjpy_lag1', 'return_usdjpy_lag2', 'return_usdjpy_lag3',
    'return_eurusd_lag1', 'return_eurusd_lag2', 'return_eurusd_lag3',
    'return_gbpusd_lag1','return_gbpusd_lag2','return_gbpusd_lag3',
    'return_ir_jpy_lag1','return_ir_jpy_lag2','return_ir_jpy_lag3',
    'return_ir_usd_lag1','return_ir_usd_lag2','return_ir_usd_lag3',
    'return_stock_us_lag1','return_stock_us_lag2','return_stock_us_lag3',
    'return_stock_jp_lag1','return_stock_jp_lag2', 'return_stock_jp_lag3']]



y = df['return_usdjpy']
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, shuffle=False)
lasso = Lasso(alpha=0.01)
lasso.fit(X_train, y_train)


predict = lasso.predict(X_test)
print('score InSample', metrics.accuracy_score(np.sign(lasso.predict(X_train)), np.sign(y_train)))
print('score OutOfSample', metrics.accuracy_score(np.sign(lasso.predict(X_test)), np.sign(y_test)))

coeff = pd.DataFrame({'x': X.columns, 'y': lasso.coef_})
coeff.set_index('x').plot.bar()


position = np.sign(lasso.predict(X_test))
v1, v2, v3, dX = train_test_split(X, df['fx_usdjpy'].diff(), test_size=0.2, shuffle=False)

strategy = position * dX
result = pd.DataFrame({'strategy': strategy,  'return': dX})
np.cumsum(result).plot()


#%%
# lasso to direction
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso

X = df_std[[
    'return_usdjpy_lag1', 'return_usdjpy_lag2', 'return_usdjpy_lag3',
    'return_eurusd_lag1', 'return_eurusd_lag2', 'return_eurusd_lag3',
    'return_gbpusd_lag1','return_gbpusd_lag2','return_gbpusd_lag3',
    'return_audusd_lag1','return_audusd_lag2','return_audusd_lag3',
    'return_oil_lag1','return_oil_lag2','return_oil_lag3',
    'return_ir_jpy_lag1','return_ir_jpy_lag2','return_ir_jpy_lag3',
    'return_ir_usd_lag1','return_ir_usd_lag2','return_ir_usd_lag3',
    'return_stock_us_lag1','return_stock_us_lag2','return_stock_us_lag3',
    'return_stock_jp_lag1','return_stock_jp_lag2', 'return_stock_jp_lag3']]



y = np.sign(df['return_usdjpy'])
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, shuffle=False)
lasso = Lasso(alpha=0.01)
lasso.fit(X_train, y_train)


predict = lasso.predict(X_test)
print('score InSample', metrics.accuracy_score(np.sign(lasso.predict(X_train)), y_train))
print('score OutOfSample', metrics.accuracy_score(np.sign(lasso.predict(X_test)), y_test))

coeff = pd.DataFrame({'x': X.columns, 'y': lasso.coef_})
coeff.set_index('x').plot.bar()


position = np.sign(lasso.predict(X_test))
v1, v2, v3, dX = train_test_split(X, df['fx_usdjpy'].diff(), test_size=0.2, shuffle=False)

strategy = position * dX
result = pd.DataFrame({'strategy': strategy,  'return': dX})
np.cumsum(result).plot()




#%%
# Lasso GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

X = df_std[[
    'return_usdjpy_lag1',
    'return_eurusd_lag1',
    'return_gbpusd_lag1',
    'return_audusd_lag1',
    'return_nzdusd_lag1',
    'return_ir_jpy_lag1',
    'return_ir_usd_lag1',
    'return_oil_lag1',
    'return_stock_us_lag1',
    'return_stock_jp_lag1']]


y = np.sign(df_std['return_usdjpy'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

y.tail()
from sklearn.linear_model import Lasso

param_grid = {'alpha': [1e-1, 1e-2, 1e-3, 1e-8, 1e-9, 1e-10, 1e-11]}
lasso = Lasso(random_state=0, max_iter=10000)
scores = ['precision', 'recall']

grid_search = GridSearchCV(lasso, param_grid, cv=5, refit=True)
coeff = grid_search.best_estimator_.coef_
coeff = pd.DataFrame({'x': X.columns, 'y': coeff})
coeff.set_index('x').plot.bar()
#print(grid_search.grid_scores_)

print(grid_search.score(X_test, y_test))
print(grid_search.best_params_)
print(grid_search.best_score_)
print(metrics.classification_report(np.sign(y_test), np.sign(predict)))

#%%
# Decision Tree GridsearchCV
from sklearn import tree
from sklearn.model_selection import GridSearchCV

X = df_std[[
    'return_usdjpy_lag1',
    'return_eurusd_lag1',
    'return_gbpusd_lag1',
    'return_audusd_lag1',
    'return_oil_lag1',
    'return_ir_jpy_lag1',
    'return_ir_usd_lag1',
    'return_stock_us_lag1',
    'return_stock_jp_lag1']]



hyper_params = {
    'random_state': [999],
    'min_samples_leaf': [3, 5, 10, 15, 20],
    'min_samples_split': [3, 5, 10, 15, 20],
    'max_depth': [3, 5, 10, 15, 20]
}
y = np.sign(df['return_usdjpy'])
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, shuffle=False)

clf_tree = GridSearchCV(tree.DecisionTreeClassifier(),           # 対象の機械学習モデル
                  hyper_params,   # 探索パラメタ辞書
                  cv=5,            # クロスバリデーションの分割数
                  verbose=False,
                  refit=True)       
clf_tree.fit(X_train, y_train)
#%%
from sklearn.tree import plot_tree
plot_tree(clf_tree.best_estimator_, filled=True, feature_names=X_train.columns)

#%%
importance = clf_tree.best_estimator_.feature_importances_
pd.Series(importance, index = X_train.columns).plot.bar()
print(clf_tree.best_params_)

predict = clf_tree.best_estimator_.predict(X_test)
position = predict
v1, v2, v3, dX = train_test_split(X, df['fx_usdjpy'].diff(), test_size=0.2, shuffle=False)

strategy = position * dX
result = pd.DataFrame({'strategy': strategy,  'return': dX})
np.cumsum(result).plot()

pd.DataFrame({
   'InSample': [clf_tree.best_estimator_.score(X_train, y_train)],
   'OutOfSample':[clf_tree.best_estimator_.score(X_test, y_test)] }, index={'score'})

#%%
# Random forest GridsearchCV
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import GridSearchCV

X = df_std[[
    'return_usdjpy_lag1',
    'return_eurusd_lag1',
    'return_gbpusd_lag1',
    'return_audusd_lag1',
    'return_oil_lag1',
    'return_ir_jpy_lag1',
    'return_ir_usd_lag1',
    'return_stock_us_lag1',
    'return_stock_jp_lag1']]



hyper_params = {
    'random_state': [999],
    'min_samples_split': [3, 5, 10, 15, 20],
    'max_depth': [3, 5, 10, 15, 20]
}
y = np.sign(df['return_usdjpy'])
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, shuffle=False)

clf_rf = GridSearchCV(RFC(),           # 対象の機械学習モデル
                  hyper_params,   # 探索パラメタ辞書
                  cv=5,            # クロスバリデーションの分割数
                  verbose=False,
                  refit=True)       
clf_rf.fit(X_train, y_train)

#%%
importance = clf_rf.best_estimator_.feature_importances_
pd.Series(importance, index = X_train.columns).plot.bar()
print(clf_rf.best_params_)

predict = clf_rf.best_estimator_.predict(X_test)
position = predict
v1, v2, v3, dX = train_test_split(X, df['fx_usdjpy'].diff(), test_size=0.2, shuffle=False)

strategy = position * dX
result = pd.DataFrame({'strategy': strategy,  'return': dX})
np.cumsum(result).plot()

pd.DataFrame({
   'InSample': [clf_rf.best_estimator_.score(X_train, y_train)],
   'OutOfSample':[clf_rf.best_estimator_.score(X_test, y_test)] }, index={'score'})


#%%
print('score InSample', clf_rf.best_estimator_.score(X_train, y_train))
print('score OutOfSample', clf_rf.best_estimator_.score(X_test, y_test))

print('test score', clf_rf.score(X_test, y_test))
print(clf_rf.best_params_)
print(clf_rf.best_score_)
predict = clf_rf.best_estimator_.predict(X_test)
print(metrics.classification_report(np.sign(y_test), np.sign(predict)))


position = predict
v1, v2, v3, dX = train_test_split(X, df['fx_usdjpy'].diff(), test_size=0.2, shuffle=False)

strategy = position * dX
result = pd.DataFrame({'strategy': strategy,  'return': dX})
np.cumsum(result).plot()


#%%
# SVM GridserchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

#X = df_std[[
#    'return_usdjpy_lag1',
#    'return_eurusd_lag1',
#    'return_gbpusd_lag1',
#    'return_ir_jpy_lag1',
#    'return_ir_usd_lag1',
#    'return_stock_us_lag1',
#    'return_stock_jp_lag1']]

X = df_std[[
    'return_usdjpy_lag1', 'return_usdjpy_lag2', 'return_usdjpy_lag3',
    'return_eurusd_lag1', 'return_eurusd_lag2', 'return_eurusd_lag3',
    'return_gbpusd_lag1','return_gbpusd_lag2','return_gbpusd_lag3',
    'return_ir_jpy_lag1','return_ir_jpy_lag2','return_ir_jpy_lag3',
    'return_ir_usd_lag1','return_ir_usd_lag2','return_ir_usd_lag3',
    'return_stock_us_lag1','return_stock_us_lag2','return_stock_us_lag3',
    'return_stock_jp_lag1','return_stock_jp_lag2', 'return_stock_jp_lag3']]

X = df_std[[
    'return_usdjpy_lag1',
    'return_eurusd_lag1',
    'return_gbpusd_lag1',
    'return_ir_jpy_lag1',
    'return_ir_usd_lag1',
    'return_stock_us_lag1',
    'return_stock_jp_lag1']]



y = np.sign(df['return_usdjpy'])
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, shuffle=False)



scores = ['precision', 'recall']
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100]},
                    {'kernel': ['linear'], 'C': [1, 10, 100]}]
                    
                    
#%%
clf_svc = GridSearchCV(SVC(), param_grid=tuned_parameters, cv=5, verbose=False, refit=True, n_jobs=-1)
clf_svc.fit(X_train, y_train)

#%%
print('train score', clf_svc.score(X_train, y_train))
print('test score', clf_svc.score(X_test, y_test))
print(clf_svc.best_params_)
print(clf_svc.best_score_)
predict = clf_svc.best_estimator_.predict(X_test)
print(metrics.classification_report(np.sign(y_test), np.sign(predict)))


position = predict
v1, v2, v3, dX = train_test_split(X, df['fx_usdjpy'].diff(), test_size=0.2, shuffle=False)

pl = position * dX
result = pd.DataFrame({'strategy': pl,  'return': dX})
np.cumsum(result).plot()



#%%
# NeuralNetwork Regress 
from sklearn.neural_network import MLPClassifier

# ハイパーパラメータのリスト
tuned_parameters = [
    {
        # 最適化手法
        "solver":['sgd'], 
        'activation': ['logistic', 'relu'],
        # 隠れ層の層の数と、各層のニューロンの数
        "hidden_layer_sizes":[(100, 10), (100, 100, 10),(100, 10, 100), (100, 10, 100, 10)], 
        "random_state":[10],
        "max_iter":[10000]
    }
]

X = df_std[[
    'return_usdjpy_lag1',
    'return_eurusd_lag1',
    'return_gbpusd_lag1',
    'return_ir_jpy_lag1',
    'return_ir_usd_lag1',
    'return_stock_us_lag1',
    'return_stock_jp_lag1']]

X = df_std[[
    'return_usdjpy_lag1',
    'return_eurusd_lag1',
    'return_gbpusd_lag1',
    'return_audusd_lag1',
    'return_oil_lag1',
    'return_ir_jpy_lag1',
    'return_ir_usd_lag1',
    'return_stock_us_lag1',
    'return_stock_jp_lag1']]



y = np.sign(df['return_usdjpy'])
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, shuffle=False)


#%%
# ニューラルネットワークの分類器とハイパーパラメータのリストを定義
clf_nn=GridSearchCV(MLPClassifier(early_stopping=True), param_grid=tuned_parameters, scoring="accuracy", cv=5, refit=True, n_jobs=-1)
clf_nn.fit(X_train, y_train)



#%%
print('train score', clf_nn.score(X_train, y_train))
print('test score', clf_nn.score(X_test, y_test))

print(clf_nn.best_params_)
print(clf_nn.best_score_)
predict = clf_nn.best_estimator_.predict(X_test)
print(metrics.classification_report(np.sign(y_test), np.sign(predict)))


position = predict
v1, v2, v3, dX = train_test_split(X, df['fx_usdjpy'].diff(), test_size=0.2, shuffle=False)

pl = position * dX
result = pd.DataFrame({'strategy': pl,  'return': dX})
np.cumsum(result).plot()


#%%
print('test score', clf_nn.score(X_test, y_test))
print(clf_nn.best_params_)
print(clf_nn.best_score_)
predict = clf_nn.best_estimator_.predict(X_test)
print(metrics.classification_report(np.sign(y_test), np.sign(predict)))

#model=MPLClassifier(solver='lbfgs', alpha=1e-8,
#    hidden_layer_sizes=5*[100], random_state=1)


pl = predict * y_test
print(pl)
result = pd.DataFrame({'pl': pl,  'return': y_test})
np.exp(np.cumsum(result)).plot()





#%%
# NeuralNetwork Regress 
from sklearn.neural_network import MLPRegressor 
model=MLPRegressor(solver='lbfgs', alpha=1e-8,
    hidden_layer_sizes=5*[100], random_state=1)

predict = model.fit(x_train, y_train).predict(X_test)

ud1 = np.sign(predict)
pl1 = ud1 * y_test
result = pd.DataFrame({'pl1': pl1,  'return': y_test})
np.exp(np.cumsum(result)).plot()

#%%
cols

#%%
# lasso 

from sklearn.model_selection import train_test_split
x_train, X_test, y_train, y_test = train_test_split(df_std[cols], df_std['return_usdjpy'], test_size=0.2, shuffle=False)

from sklearn.linear_model import Lasso
model = Lasso(alpha=1e-3)
predict = model.fit(x_train, y_train).predict(X_test)
print(model.coef_)
coeff = pd.DataFrame({'x': cols, 'y': model.coef_})
coeff.set_index('x').plot.bar()

ud1 = np.sign(predict)
pl1 = ud1 * y_test
result = pd.DataFrame({'pl1': pl1,  'return': y_test})
np.exp(np.cumsum(result)).plot()

#%%
from sklearn.model_selection import GridSearchCV
param_grid = {'alpha': [1, 0.5, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-9, 1e-10, 1e-11]}
model = Lasso()
grid_search = GridSearchCV(model, param_grid, cv=5, return_train_score=True)
grid_search.fit(x_train, y_train)

print(grid_search.best_estimator_.score(x_train, y_train))
print(grid_search.best_score_)

#%%
from sklearn.neural_network import MLPRegressor 
model=MLPRegressor(solver='lbfgs', alpha=1e-8,
    hidden_layer_sizes=5*[100], random_state=1)

predict = model.fit(x_train, y_train).predict(X_test)

ud1 = np.sign(predict)
pl1 = ud1 * y_test
result = pd.DataFrame({'pl1': pl1,  'return': y_test})
np.exp(np.cumsum(result)).plot()

#%%
df.info()

#%%
# learning curve
from sklearn.model_selection import learning_curve
X = df[cols]
y = df['return_usdjpy']

estimator = model   
train_sizes = np.arange(100, 2000, 500)
train_sizes, train_scores, test_scores = learning_curve(
    estimator, X, y, cv=3, train_sizes=train_sizes, random_state=42, shuffle=False)

#%%
print(train_sizes)
print(train_scores)
print(test_scores)



#%%
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

print(train_scores_mean)
print(train_scores_std)
print(test_scores_mean)
print(test_scores_std)

import matplotlib.pyplot as plt

plt.figure()
plt.title("Learning Curve")
plt.xlabel("Training examples")
plt.ylabel("Score")

# Traing score と Test score をプロット
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Test score")

# 標準偏差の範囲を色付け
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, color="r", alpha=0.2)
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, color="g", alpha=0.2)

plt.legend(loc="best")
plt.show()


# %%

#%%
import numpy as np
import pandas as pd
import datetime as dt
from pylab import mpl, plt
import pandas_datareader.data as web
plt.style.use('seaborn')
%matplotlib inline

start='2011/3/26'
fx_usdjpy = web.DataReader('DEXJPUS', 'fred',start).interpolate()
fx_gbpusd = web.DataReader('DEXUSUK', 'fred', start).interpolate()
fx_audusd = web.DataReader('DEXUSAL', 'fred', start).interpolate()
fx_nzdusd = web.DataReader('DEXUSNZ', 'fred', start).interpolate()

bond_jp = web.DataReader('IRLTLT01JPM156N', 'fred', start).interpolate()
bond_us = web.DataReader('DGS10','fred', start).interpolate()
swap_us = web.DataReader('DSWP10','fred', start).interpolate()

#%%
df = pd.DataFrame({
    'fx_usdjpy': web.DataReader('DEXUSUK', 'fred', start).interpolate()['DEXUSUK'],
    'fx_audusd': web.DataReader('DEXUSAL', 'fred', start).interpolate()['DEXUSAL'],
    'fx_nzdusd': web.DataReader('DEXUSNZ', 'fred', start).interpolate()['DEXUSNZ'],
    'jpy_libor': web.DataReader('JPY3MTD156N','fred', start).interpolate()['JPY3MTD156N'],
    'usd_libor': web.DataReader('USD3MTD156N','fred', start).interpolate()['USD3MTD156N'],
    'oil': web.DataReader('DCOILBRENTEU', 'fred', start).interpolate()['DCOILBRENTEU']
})
print(df)


#%%
f = df[df.index > dt.datetime(2016,4,1)]
f[f.index < dt.datetime(2017,4,1)].plot()

# %%
# correlation AUD & NZD
df['aud_logdiff'] = np.log(df['fx_audusd']).diff()
df['nzd_logdiff'] = np.log(df['fx_nzdusd']).diff()
df.plot.scatter(x='aud_logdiff', y='nzd_logdiff')

#%%
# correlation AUD & OIL
df['oil_logdiff'] = np.log(df['oil']).diff()
df.plot.scatter(x='oil_logdiff', y='aud_logdiff')

#%%
# correlation FX & IR diff
df['jpy_logdiff'] = np.log(df['fx_usdjpy']).diff()
df['ir_usdjpy_diff'] = (df['jpy_libor']).diff() /  df['jpy_libor'] - (df['usd_libor']).diff() /  df['usd_libor']
df.plot.scatter(x='ir_usdjpy_diff', y='jpy_logdiff')

#%%
sma_short = 20
sma_long = 200

data = pd.DataFrame({
    'fx_usdjpy': df['fx_usdjpy'],
    'SMA_short': df['fx_usdjpy'].rolling(sma_short).mean(),
    'SMA_long': df['fx_usdjpy'].rolling(sma_long).mean()
})
data.plot(figsize=(10, 6))

#%%
df = pd.DataFrame({
    'fx': np.log(fx).diff()['DEXJPUS'],
    'n225': np.log(n225).diff()['NIKKEI225'],
    'n100': np.log(n100).diff()['NASDAQ100'],
    'j1': j1.diff()['JPY12MD156N']}).interpolate()[1:]

# %%
