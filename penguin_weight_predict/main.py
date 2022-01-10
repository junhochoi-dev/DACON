import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression

url_data = 'C:/Users/CHOI JUN HO/Desktop/Dacon/penguin_weight_predict/00_data/'

df_train = pd.read_csv(url_data + 'train.csv') # 113 samples
df_testcase = pd.read_csv(url_data + 'test.csv') # 227 testcase

### Yes or No -> True and False
df_train = df_train.replace('Yes', True)
df_train = df_train.replace('No', False)
df_testcase = df_testcase.replace('Yes', True)
df_testcase = df_testcase.replace('No', False)

### Male or Female -> True and False
df_train = df_train.replace('MALE', True)
df_train = df_train.replace('FEMALE', False)
df_testcase = df_testcase.replace('MALE', True)
df_testcase = df_testcase.replace('FEMALE', False)

### DATA PREPROCESSING # ONE HOT ENCODING
# pd.get_dummies(df_train['Species'], columns=['Species'], prefix='Species')
Species_list = [
    'Adelie Penguin (Pygoscelis adeliae)',
    'Chinstrap penguin (Pygoscelis antarctica)',
    'Gentoo penguin (Pygoscelis papua)'
    ]
Island_list = ['Biscoe', 'Dream', 'Torgersen']
for species in Species_list:
    df_train['Species_'+f'{species}'] = df_train['Species'] == species
    df_testcase['Species_'+f'{species}'] = df_testcase['Species'] == species
for island in Island_list:
    df_train['Island_'+f'{island}'] = df_train['Island'] == island
    df_testcase['Island_'+f'{island}'] = df_testcase['Island'] == island

# column does not have white_space
df_train.columns = df_train.columns.str.replace(' ', '_')
df_testcase.columns = df_testcase.columns.str.replace(' ', '_')

### is there null value?
origin = df_train #*****
origin_t = df_testcase #*****
df_train.replace(' ', np.nan, inplace=True)
df_testcase.replace(' ', np.nan, inplace=True)
df_train = df_train.dropna()
# print(df_train['Species'].value_counts())

###
# penguin_input = df_train.loc[:, df_train.columns.difference(['id', 'Species', 'Island', 'Body_Mass_(g)'])]
# penguin_target = df_train[['Body_Mass_(g)']]
# df_testcase = df_testcase.loc[:, df_testcase.columns.difference(['id', 'Species', 'Island'])]

### train, test split
# train_input, test_input, train_target, test_target = train_test_split(penguin_input, penguin_target, train_size=0.7, test_size=0.3, random_state=alpha)

### Train id=6 Change Value Sex nan -> True(MALE)
### Train id=70 Change Value Sex nan -> False(FEMALE)
# penguin_input = df_train.loc[:, df_train.columns.difference(['id', 'Species', 'Island', 'Sex'])]
# penguin_target = df_train[['Sex']]
# lr = LinearRegression()
# model = lr.fit(penguin_input, penguin_target)
# origin = origin.loc[:, origin.columns.difference(['id', 'Species', 'Island', 'Sex'])]
# t = origin.loc[[6, 70]]
# print(model.predict(t))

### Train id=8 Change Value Sex nan -> False(FEMALE)
# penguin_input = df_train.loc[:, df_train.columns.difference(['id', 'Species', 'Island', 'Sex', 'Delta_13_C_(o/oo)', 'Delta_15_N_(o/oo)'])]
# penguin_target = df_train[['Sex']]
# lr = LinearRegression()
# model = lr.fit(penguin_input, penguin_target)
# origin = origin.loc[:, origin.columns.difference(['id', 'Species', 'Island', 'Sex', 'Delta_13_C_(o/oo)', 'Delta_15_N_(o/oo)'])]
# t = origin.loc[[8]]
# print(model.predict(t))

### Train id=8 Change Value Delta_13_C_(o/oo) nan -> -26.17444
### Train id=18 Change Value Delta_13_C_(o/oo) nan -> -25.27259
### Train id=109 Change Value Delta_13_C_(o/oo) nan -> -25.87942
# penguin_input = df_train.loc[:, df_train.columns.difference(['id', 'Species', 'Island', 'Delta_13_C_(o/oo)', 'Delta_15_N_(o/oo)'])]
# penguin_target = df_train[['Delta_13_C_(o/oo)']]
# lr = LinearRegression()
# model = lr.fit(penguin_input, penguin_target)
# origin = origin.loc[:, origin.columns.difference(['id', 'Species', 'Island', 'Delta_13_C_(o/oo)', 'Delta_15_N_(o/oo)'])]
# t = origin.loc[[8, 18, 109]]
# print(model.predict(t))

### Train id=8 Change Value Delta_15_N_(o/oo) nan -> 8.50302
### Train id=18 Change Value Delta_15_N_(o/oo) nan -> 9.02920
### Train id=109 Change Value Delta_15_N_(o/oo) nan -> 8.45511
# penguin_input = df_train.loc[:, df_train.columns.difference(['id', 'Species', 'Island', 'Delta_15_N_(o/oo)'])]
# penguin_target = df_train[['Delta_15_N_(o/oo)']]
# lr = LinearRegression()
# model = lr.fit(penguin_input, penguin_target)
# origin = origin.loc[:, origin.columns.difference(['id', 'Species', 'Island', 'Delta_15_N_(o/oo)'])]
# t = origin.loc[[8, 18, 109]]
# print(model.predict(t))

### Test id=46 Change Value Sex nan -> False(FEMALE)
### Test id=98 Change Value Sex nan -> False(FEMALE)
### Test id=152 Change Value Sex nan -> False(FEMALE)
### Test id=209 Change Value Sex nan -> False(FEMALE)
# penguin_input = df_train.loc[:, df_train.columns.difference(['id', 'Species', 'Island', 'Sex', 'Body_Mass_(g)'])]
# penguin_target = df_train[['Sex']]
# lr = LinearRegression()
# model = lr.fit(penguin_input, penguin_target)
# origin_t = origin_t.loc[:, origin_t.columns.difference(['id', 'Species', 'Island', 'Sex'])]
# t = origin_t.loc[[46, 98, 152, 209]]
# print(model.predict(t))

### Test id=75 Change Value Delta_15_N_(o/oo) nan -> 9.56700
# penguin_input = df_train.loc[:, df_train.columns.difference(['id', 'Species', 'Island', 'Delta_15_N_(o/oo)', 'Body_Mass_(g)'])]
# penguin_target = df_train[['Delta_15_N_(o/oo)']]
# lr = LinearRegression()
# model = lr.fit(penguin_input, penguin_target)
# origin_t = origin_t.loc[:, origin_t.columns.difference(['id', 'Species', 'Island', 'Delta_15_N_(o/oo)'])]
# t = origin_t.loc[[75]]
# print(model.predict(t))

### Test id=81 Change Value Sex nan -> False(FEMALE)
### Test id=205 Change Value Sex nan -> False(FEMALE)
# penguin_input = df_train.loc[:, df_train.columns.difference(['id', 'Species', 'Island', 'Sex', 'Delta_13_C_(o/oo)', 'Delta_15_N_(o/oo)', 'Body_Mass_(g)'])]
# penguin_target = df_train[['Sex']]
# lr = LinearRegression()
# model = lr.fit(penguin_input, penguin_target)
# origin_t = origin_t.loc[:, origin_t.columns.difference(['id', 'Species', 'Island', 'Sex', 'Delta_13_C_(o/oo)', 'Delta_15_N_(o/oo)', 'Body_Mass_(g)'])]
# t = origin_t.loc[[81, 205]]
# print(model.predict(t))

### Test id=[27, 81, 106, 159, 175, 202, 205, 215] Change Value Delta_13_C_(o/oo)
# [[-25.81132302][-25.33797789][-26.43561483][-25.75198031][-25.83742751][-25.81681988][-25.82890281][-25.44691295]]
# penguin_input = df_train.loc[:, df_train.columns.difference(['id', 'Species', 'Island', 'Delta_13_C_(o/oo)', 'Delta_15_N_(o/oo)', 'Body_Mass_(g)'])]
# penguin_target = df_train[['Delta_13_C_(o/oo)']]
# lr = LinearRegression()
# model = lr.fit(penguin_input, penguin_target)
# origin_t = origin_t.loc[:, origin_t.columns.difference(['id', 'Species', 'Island', 'Delta_13_C_(o/oo)', 'Delta_15_N_(o/oo)', 'Body_Mass_(g)'])]
# t = origin_t.loc[[27, 81, 106, 159, 175, 202, 205, 215]]
# print(model.predict(t))

### Test id=[27, 81, 106, 159, 175, 202, 205, 215] Change Value Delta_15_N_(o/oo)
# [[8.70580334][8.86333507][8.38509087][8.47212078][9.06540768][8.70221055][8.48558182][9.06811045]]
# penguin_input = df_train.loc[:, df_train.columns.difference(['id', 'Species', 'Island', 'Delta_15_N_(o/oo)', 'Body_Mass_(g)'])]
# penguin_target = df_train[['Delta_15_N_(o/oo)']]
# lr = LinearRegression()
# model = lr.fit(penguin_input, penguin_target)
# origin_t = origin_t.loc[:, origin_t.columns.difference(['id', 'Species', 'Island', 'Delta_15_N_(o/oo)', 'Body_Mass_(g)'])]
# t = origin_t.loc[[27, 81, 106, 159, 175, 202, 205, 215]]
# print(model.predict(t))

### GET WEIGHT!!!
# penguin_input = df_train.loc[:, df_train.columns.difference(['id', 'Species', 'Island', 'Body_Mass_(g)'])]
# penguin_target = df_train[['Body_Mass_(g)']]
# lr = LinearRegression()
# model = lr.fit(penguin_input, penguin_target)
# origin_t = origin_t.loc[:, origin_t.columns.difference(['id', 'Species', 'Island', 'Body_Mass_(g)'])]
# df = pd.DataFrame(model.predict(origin_t))
# df.to_csv('result.csv', index=True)

### GET WEIGHT!!!
penguin_input = df_train.loc[:, df_train.columns.difference(['id', 'Species', 'Island', 'Body_Mass_(g)'])]
penguin_target = df_train[['Body_Mass_(g)']]
df_testcase = df_testcase.loc[:, df_testcase.columns.difference(['id', 'Species', 'Island'])]

# random_state = 3 8
train_input, test_input, train_target, test_target = train_test_split(penguin_input, penguin_target, train_size=0.7, test_size=0.3, random_state=8980)
# lr = LinearRegression()
# lr.fit(train_input, train_target)
# print(lr.score(train_input, train_target))
# print(lr.score(test_input, test_target))

poly = PolynomialFeatures(degree=1, include_bias=False) # 1 > 2 > 3 ...
poly.fit(penguin_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)
testcase_poly = poly.transform(df_testcase)

lr = LinearRegression()
lr.fit(train_poly, train_target)
print('##### LinearRegression')
print(lr.score(train_poly, train_target))
print(lr.score(test_poly, test_target))
print(lr.predict(testcase_poly))
df = pd.DataFrame(lr.predict(testcase_poly))
df.to_csv('result.csv', index=True)

# ss = StandardScaler()
# ss.fit(train_poly)
# train_scaled = ss.transform(train_poly)
# test_scaled = ss.transform(test_poly)

### RIDGE REGRESSION
# from sklearn.linear_model import Ridge
# train_score = []
# test_score = []
#
# alpha_list = [0.001, 0.01, 0.1, 1, 10, 100] # 0.01
# for alpha in alpha_list:
#     ridge = Ridge(alpha=alpha)
#     ridge.fit(train_scaled, train_target)
#     train_score.append(ridge.score(train_scaled, train_target))
#     test_score.append(ridge.score(test_scaled, test_target))
#
# plt.plot(np.log10(alpha_list), train_score)
# plt.plot(np.log10(alpha_list), test_score)
# plt.xlabel('alpha')
# plt.ylabel('R^2')
# plt.show()
#
# ridge = Ridge(alpha=0.1)
# print('##### RidgeRegression')
# model = ridge.fit(train_scaled, train_target)
# print(ridge.score(train_scaled, train_target))
# print(ridge.score(test_scaled, test_target))
# print(model.predict(testcase_poly))
# df = pd.DataFrame(model.predict(testcase_poly))
# df.to_csv('result.csv', index=True)

### LASSO REGRESSION
# from sklearn.linear_model import Lasso
# lasso = Lasso()
# lasso.fit(train_scaled, train_target)
# train_score = []
# test_score = []
#
# alpha_list = [0.001, 0.01, 0.1, 1, 10, 100] # 0.01
# for alpha in alpha_list:
#     lasso = Lasso(alpha=alpha)
#     lasso.fit(train_scaled, train_target)
#     train_score.append(lasso.score(train_scaled, train_target))
#     test_score.append(lasso.score(test_scaled, test_target))
#
# plt.plot(np.log10(alpha_list), train_score)
# plt.plot(np.log10(alpha_list), test_score)
# plt.xlabel('alpha')
# plt.ylabel('R^2')
# plt.show()

# RMSE

# Body_Mass_(g)  Clutch_Completion      Culmen_Depth_(mm)    Culmen_Length_(mm)
# Delta_13_C_(o/oo)    Delta_15_N_(o/oo)
# Flipper_Length_(mm)
# Island_Biscoe    Island_Dream    Island_Torgersen
# Species_Adelie_Penguin_(Pygoscelis_adeliae)    Species_Chinstrap_penguin_(Pygoscelis_antarctica)    Species_Gentoo_penguin_(Pygoscelis_papua)

# print(test_input)
# print(lr.predict(test_input))

# print(df_train)
# print(tabulate(df_train, headers="keys"))
#ipython

# sns.boxplot(x='Island', y='Body Mass (g)', hue='Species', data=df_penguin)
# sns.swarmplot(x='Island', y='Body Mass (g)', hue='Species', data=df_penguin, color='.25')
# plt.show()

# < penguin species >
# 1. Adelie Penguin (Pygoscelis adeliae)
# 2. Chinstrap penguin (Pygoscelis antarctica)
# 3. Gentoo penguin (Pygoscelis papua)

# condition
# Island -> Biscoe, Dream, Torgersen
# Clutch completion -> Yes, No
# Sex -> MALE, FEMALE

# NULL 값 존재
# 1. 그 training sample 없애보고 시작하자...