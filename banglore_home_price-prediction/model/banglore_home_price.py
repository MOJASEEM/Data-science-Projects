import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
import matplotlib
import pickle
import json

matplotlib.rcParams['figure.figsize'] = (20,10)

df1 = pd.read_csv("D:/Data Science/banglore_home_price_prediction/bengaluru_house_prices.csv")
print(df1.groupby('area_type')['area_type'].agg('count'))
df2 = df1.drop(['area_type', 'society', 'balcony', 'availability'], axis='columns')
print(df2.isnull().sum())
df3 = df2.dropna()
print(df3.isnull().sum())
print(df3['size'].unique())
df3=df3.copy()
df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))
print(df3['bhk'].unique())
print(df3[df3.bhk>20])
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True
print(df3[~df3['total_sqft'].apply(is_float)].head())
def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1])) / 2
    try:
        return float(x)
    except:
        return None
df4 = df3.copy()
df4['total_sqft'] = df4['total_sqft'].apply(convert_sqft_to_num)
print(df4.loc[30])
df5=df4.copy()
df5['price_per_sqft'] = df5['price']*100000/df5['total_sqft']
print(df5.location.unique())
print(len(df5.location.unique()))
df5.location = df5.location.apply(lambda x: x.strip())
location_stats = df5.groupby('location')['location'].agg('count').sort_values(ascending=False)
print(location_stats)
less_than_10 = location_stats[location_stats<=10]
print(less_than_10)
df5.location = df5.location.apply(lambda x: 'other' if x in less_than_10 else x)
print(len(df5.location.unique()))
df6 = df5[~(df5.total_sqft/df5.bhk<300)]
df6.price_per_sqft.describe()
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out
df7 = remove_pps_outliers(df6)
df7.shape
def plot_scatter_chart(df, location):
    bhk2 = df[(df.location==location) & (df.bhk==2)]
    bhk3 = df[(df.location==location) & (df.bhk==3)]
    matplotlib.rcParams['figure.figsize'] = (15,10)
    plt.scatter(bhk2.total_sqft, bhk2.price, color='blue', label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft, bhk3.price, marker='+', color='green', label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price Per Square Feet")
    plt.title(location)
    plt.legend()
    plt.show()
plot_scatter_chart(df7, "Rajaji Nagar")
def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count'] > 5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft < (stats['mean'])].index.values)
    return df.drop(exclude_indices, axis='index')
df8 = remove_bhk_outliers(df7)
print(df8.shape)
plot_scatter_chart(df8, "Rajaji Nagar")
plt.hist(df8.price_per_sqft, rwidth=0.8)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")
plt.show()
df8.bath.unique()
df9 = df8[df8.bath<df8.bhk+2]
print(df9.shape)
df10 = df9.drop(['price_per_sqft'], axis='columns')
dummies = pd.get_dummies(df10.location)
df11 = pd.concat([df10, dummies.drop('other', axis='columns')], axis='columns')
df12 = df11.drop(['location','size'], axis='columns')
X = df12.drop('price', axis='columns')
y = df12.price
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
lr_clf = LinearRegression()
lr_clf.fit(x_train, y_train)
print(lr_clf.score(x_test, y_test))
cv=ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
cross_val_score(LinearRegression(), X, y, cv=cv)
def find_best_model_using_gridsearchcv(X, y):
    algos = {
        'linear_regression': {
            'model': LinearRegression(),
            'params': {
                'copy_X': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1, 2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion': ['squared_error', 'friedman_mse'],
                'splitter': ['best', 'random']
            }
        }

    }
    scores = []
    cv=ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs = GridSearchCV(config['model'], config['params'], cv=5, return_train_score=False)
        gs.fit(X, y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })
    return pd.DataFrame(scores)
find_best_model_using_gridsearchcv( X, y)
def predict_price(location, sqft, bath, bhk):
    loc_index = np.where(X.columns==location)[0][0]
    
    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if location in X.columns:
        loc_index = X.columns.get_loc(location)
        x[loc_index] = 1
    
    x_df = pd.DataFrame([x], columns=X.columns)

    return lr_clf.predict(x_df)[0]
print(predict_price('1st Phase JP Nagar', 1000, 2, 2))
print(predict_price('Indira Nagar', 1000, 2, 2))
print(predict_price('Indira Nagar', 1000, 3, 3))
with open('D:/Data Science/banglore_home_price_prediction/model/banglore_home_price_model.pickle', 'wb') as f:
    pickle.dump(lr_clf, f) 
columns = {
    'data_columns' : [col.lower() for col in X.columns]
}
with open("D:/Data Science/banglore_home_price_prediction/model/columns.json", "w+") as f:
    f.write(json.dumps(columns))
    