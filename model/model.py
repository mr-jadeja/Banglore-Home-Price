import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split,ShuffleSplit,cross_val_score,GridSearchCV
from sklearn.linear_model import LinearRegression,Lasso
import pickle
import json
from sklearn.tree import DecisionTreeRegressor

import matplotlib

df1 = pd.read_csv('dataset.csv')
df2 = df1.drop(['area_type','society','balcony','availability'],axis='columns')
#__    print(df2.isnull().sum()) it is return all null value numbers
df3 = df2.dropna()
# now all null value row drop and new dataframe is created
# print(df3['size'].unique()) it will return all the unique value


df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))
# in above line, lambda fun split string into two parts after space and after [0] means we require only one value
# now we are explore total_sqrt column so we found range like 1120-2993 so we have to take avg

def is_float(x):
    try:
        float(x)
    except:
        return False
    return True

# print(df3[~df3['total_sqft'].apply(is_float)]) by doing this we got all ranges in column because it is not float

def convert_sqrt_to_num(x):
    token = x.split('-')
    if len(token) == 2:
        return (float(token[0])+float(token[1]))/2
    try:
        return float(x)
    except:
        return None

df4 = df3.copy() #it will create copy of df3
df4['total_sqft'] = df4['total_sqft'].apply(convert_sqrt_to_num)
# print(df4.loc[30]) so it will take avg and give the ans

df5 = df4.copy()
#feature engineering will very useful for further process so now we can use to create new column
df5['price_per_sqft'] = (df5['price']*100000)/df5['total_sqft']
# print(df5.head()) so new column is added
# print(len(df5.location.unique())) it will tell us that there will be total 1304 area

location_stats = df5.groupby('location')['location'].agg('count').sort_values(ascending=False)
# print(location_stats) it will gave all the location raw in decending order
# print(len(location_stats[location_stats <= 10])) it will return all location which have less than 10 houses
location_stats_less_than_10 = location_stats[location_stats <= 10]
df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
# so all location which have less than 10 houses will be added in others
# print(len(df5.location.unique())) so now only 242 location are there

# outliner removal like house is not big like 5000 sqft so it is wrong so we have to change
# print(df5[df5.total_sqft/df5.bhk < 300].head()) it will return all false value because it will create error

df6 = df5[~(df5.total_sqft/df5.bhk < 300)]
# so it will remove many data which is not correct
# print(df6.price_per_sqrt.describe()) output:-
# count     12456.000000
# mean       6308.502826
# std        4168.127339
# min         267.829813
# 25%        4210.526316
# 50%        5294.117647
# 75%        6916.666667
# max      176470.588235 here we have to do with standrd deviation
def remove_pps_outlier(df):
    df_out = pd.DataFrame()
    for key,subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft > (m-st)) & (subdf.price_per_sqft <= (m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out

df7 = remove_pps_outlier(df6)
# it will reduced 2000 dataS

# in our dataset we have find that 2 bhk price is more than 3 bhk price so it might because of location or it might because of error
# now we can plot the graph
def plot_graph(df,location):
    bhk2 = df[(df.location == location) & (df.bhk == 2)]
    bhk3 = df[(df.location == location) & (df.bhk == 3)]
    plt.scatter(bhk2.total_sqft,bhk2.price,edgecolors='blue',label='2 BHK',s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,edgecolors='green',marker='+',label='3 BHK',s=50)
    plt.xlabel('total area')
    plt.ylabel('price')
    plt.title(location)
    plt.legend()
    plt.show()


# plot_graph(df7,'Rajaji Nagar') after running this we got exact information

# so we do using mean
def remove_bhk(df):
    exclude_indices = np.array([])
    for location,location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk,bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std' : np.std(bhk_df.price_per_sqft),
                'count':bhk_df.shape[0]
            }
        for bhk,bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count'] > 5:
                exclude_indices = np.append(exclude_indices,bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')

df8 = remove_bhk(df7)
# so it will delete many raws in dataset
# in our dataset we have 5 bedrooms so we havenot 7 or more bathroom so we have to remove this records
df9 = df8[df8.bath<df8.bhk+2]

# we have to drop many columns like size and price_per_sqft because size has bhk column and price column not useful
df10 = df9.drop(['size','price_per_sqft'],axis='columns')
# here we have to convert location text to number for training purpose
dummies = pd.get_dummies(df10.location)
df11 = pd.concat([df10,dummies.drop('other',axis='columns')],axis='columns')
df12 = df11.drop('location',axis='columns')

#-----------------------MACHINE     LEARNING          MODEL --------------------------------------------
# so now we move to the model train
x = df12.drop('price',axis='columns')
y = df12.price

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=10)

#so now we can use linear regression

lr_clf =LinearRegression()
lr_clf.fit(x_train,y_train)
#print(lr_clf.score(x_test,y_test)) #here score is nearly 80 % so it will not useful

# now we use  cross validation
#cv = ShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
#print(cross_val_score(LinearRegression(),x,y,cv=cv))

# now try different algorith for that we use gridsearch cv

def find_best_score(x,y):
    algos = {
        'linear_regression': {
            'model':LinearRegression(),
            'params': {
                'normalize': [True,False]
            }
        },
        'lasso':{
            'model':Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random','cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion': ['mse','friedman_mse'],
                'splitter': ['best','random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
    for algo_name,config in algos.items():
        gs = GridSearchCV(config['model'],config['params'],cv=cv,return_train_score=False)
        gs.fit(x,y)
        scores.append({
            'model': algo_name,
            'best_score':gs.best_score_,
            'best_params':gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

data = find_best_score(x,y)
# print(data) after that we reached at conclusion that linear regression with {'normalize': False} params are best
# now make function that will predict the price
def predict_price(location,sqft,bath,bhk):
    loc_index = np.where(x.columns == location)[0][0]

    y =np.zeros(len(x.columns))
    y[0] = sqft
    y[1] = bath
    y[2] = bhk
    if loc_index >= 0:
        y[loc_index] = 1
    return lr_clf.predict([y])[0]

# print(predict_price('Indira Nagar',1000,2,2)) so it will predict price is 187 lakhs

with open('banglore_price_model.pickle','wb') as f:
    pickle.dump(lr_clf,f)

# now we collect all columns name in file
columns = {
    'data_columns': [col.lower() for col in x.columns]
}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))