import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score from sklearn.inspection import permutation_importance
import scipy.stats as stats
from scipy.stats import f_oneway
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
# Data Processing and Cleaning
df = pd.read_csv("...") # Various Datasets
df.dropna(inplace=True)
print(df.columns)
df_num = df[[’length_ft’,’price’,’dryWeight_lb’, ’totalHP’]]
df_num.corr()
plt.figure(figsize=(16, 6))
heatmap = sns.heatmap(df.corr(), vmin=−1, vmax=1, annot=True) heatmap.set_title(’Correlation Heatmap’, fontdict={’fontsize’:12}, pad=12);
# Random Forest
# boxplot1 = df.boxplot(column=[’price’])
Q1 = df_num["price"].quantile(0.25)
Q3 = df_num["price"].quantile(0.75)
IQR = Q3 − Q1
lower_bound = Q1 − 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_num = df_num[(df_num["price"] >= lower_bound) & (df_num["price"] <= upper_bound)] X = df_num.drop(’price’, axis = 1)
y = df_num[’price’]
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.2, random_state=1000)
rf = RandomForestRegressor(n_estimators=10, max_depth=10
, random_state=1000)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", mean_squared_error(y_test, y_pred, squared=False)) boxplot = df.boxplot(column=[’price’])
boxplot2 = df_num.boxplot(column=[’price’])
residuals = y_test − y_pred
# Plot the residuals versus predicted values
plt.scatter(y_pred, residuals)
plt.xlabel("Predicted prices")
plt.ylabel("Residuals")
plt.title("Random Forest Regression Results")
plt.show()
importances = rf.feature_importances_
for i, feature_name in enumerate(X.columns):
print(f"{feature_name}: {importances[i]}")
# Adaboost Regression
adaboost_model = AdaBoostRegressor(random_state=100)
adaboost_model.fit(X_train, y_train)
y_pred = adaboost_model.predict(X_test)
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", mean_squared_error(y_test, y_pred, squared=False)) residuals = y_test − y_pred
# Plot the residuals versus predicted values
plt.scatter(y_pred, residuals)
plt.xlabel("Predicted prices")
plt.ylabel("Residuals")
plt.title("Adaboost Regression Results")
plt.show()
result = permutation_importance(adaboost_model, X, y, n_repeats=10, random_state=0) importances = result.importances_mean
for i, feature_name in enumerate(X.columns):
print(f"{feature_name}: {importances[i]}")
# Support Vector Regression
svr_model = SVR(kernel=’rbf’, C=100, gamma=0.1, epsilon=.1)
svr_model.fit(X_train, y_train)
y_pred = svr_model.predict(X_test)
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", mean_squared_error(y_test, y_pred, squared=False)) svr_model = SVR(kernel=’linear’)
svr_model.fit(X_train, y_train)
y_pred = svr_model.predict(X_test)
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", mean_squared_error(y_test, y_pred, squared=False)) importances = abs(svr_model.coef_[0])
for i, feature_name in enumerate(X.columns):
print(f"{feature_name}: {importances[i]}")
# Neural Network Regression
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation=’relu’)) model.add(Dense(32, activation=’relu’))
model.add(Dense(1))
model.compile(loss=’mean_squared_error’, optimizer=’adam’)
history = model.fit(X_train, y_train, epochs=50000,
batch_size=32, verbose=1, validation_split=0.2)
y_pred = model.predict(X_test)
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", mean_squared_error(y_test, y_pred, squared=False))
# Muliple Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", mean_squared_error(y_test, y_pred, squared=False)) print("Coefficients: \n", lr_model.coef_)
# Calculate Residuals
residuals = y_test − y_pred
# Plot the residuals versus predicted values
plt.scatter(y_pred, residuals)
plt.xlabel("Predicted prices")
plt.ylabel("Residuals")
plt.title("Multiple Linear Regression Results")
plt.show()
# F−stats and P−values
f_stat, p_value = f_oneway(*[df["price"] for name, group in df.groupby("state")]) print("F−statistic:", f_stat)
print("p−value:", p_value)
# perform post−hoc test (Tukey’s HSD)
tukey_results = pairwise_tukeyhsd(df["price"], df["state"]) print(tukey_results)
