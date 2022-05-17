""":DATA --> The key data (key.csv) indicates for each store to which weather station it belongs"""
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, \
    AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelBinarizer

""":DATA --> weather data (weather.csv) of each weather station."""

""":GOAL --> Predict how sales of weather-sensitive products are affected by snow and rain"""
# observations from the years 2012-2013 as your training
# observations from the year 2014 as your test data

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
import my_functions as mf
from statsmodels.graphics import utils
from sklearn.metrics import accuracy_score, mean_squared_error


def print_df(df):
    print(df)


""":param --> csv file"""
""":returns --> pandas data frame with csv file data"""


def csv_to_df(csv_file):
    df = pd.read_csv(csv_file)
    return df


############## SPLIT DF FUNCTIONS ##############

# split the data by year, default year is 2014
def split_data_to_test(df, year=2014):
    df['date'] = pd.to_datetime(df['date'])
    filtered_df = df[df['date'].dt.year == year]
    return filtered_df


# split the data by year, default from year 2012 to year 2013
def split_data_to_train(df, from_year=2012, to_year=2013):
    df['date'] = pd.to_datetime(df['date'])
    filtered_df = df.loc[(df['date'] >= f'{from_year}-01-01')
                         & (df['date'] <= f'{to_year}-01-1')]
    return filtered_df


############## PLOT FUNCTIONS ##############

def plot_by_date(df):
    df = df.sort_values('date', ascending=True)
    plt.plot(df['date'], df['units'])
    plt.xticks(rotation='vertical')
    plt.title('sales by date')
    plt.xlabel('date')
    plt.ylabel('units sold')
    plt.show()


def plot_by_store_numer(df):
    plt.plot(df['store_nbr'], df['units'])
    plt.xticks(rotation='vertical')
    plt.title('sales by store number')
    plt.xlabel('store number')
    plt.ylabel('units sold')
    plt.show()


def plot_by_item_number(df):
    plt.plot(df['item_nbr'], df['units'], '*')
    plt.xticks(rotation='vertical')
    plt.title('sales by item number')
    plt.xlabel('item number')
    plt.ylabel('units sold')
    plt.show()


def explore_data(train_df):
    plt.figure(figsize=(17, 6))
    plt.subplot(131)
    sns.distplot(train_df["units"])
    plt.subplot(132)
    stats.probplot(train_df["units"], plot=plt)
    plt.subplot(133)
    sns.boxplot(train_df["units"])
    plt.tight_layout()
    plt.show()


def explore_data_2(df_train):
    train_units_log1p = np.log1p(df_train["units"])
    plt.figure(figsize=(17, 6))
    plt.subplot(131)
    sns.distplot(train_units_log1p)
    plt.subplot(132)
    stats.probplot(train_units_log1p, plot=plt)
    plt.subplot(133)
    sns.boxplot(train_units_log1p)
    plt.tight_layout()
    plt.show()


def plot_missing_data(weather_data):
    weather_data_cp = weather_data.copy()
    weather_data_cp = mf.changeTypes(weather_data_cp)
    numerical_features, categorical_features = mf.divideIntoNumericAndCategoricalVariables(weather_data_cp)
    weather_data_null = weather_data_cp.isnull().sum()
    weather_data_null = weather_data_null.drop(weather_data_null[weather_data_null == 0].index).sort_values(
        ascending=False)
    weather_data_missing = pd.DataFrame({'Missing Numbers': weather_data_null})
    weather_data_null = weather_data_null / len(weather_data_cp) * 100
    # Barplot missing values
    f, ax = plt.subplots(figsize=(15, 12))
    plt.xticks(rotation='90')
    sns.barplot(x=weather_data_null.index, y=weather_data_null)
    plt.xlabel('Features', fontsize=15)
    plt.ylabel('Percent of missing values', fontsize=15)
    plt.title('Percent missing data by feature \n Overall, Depart, Sunrise, Sunset, and Snowfall contain a lot of '
              'Missing Values', fontsize=15)
    plt.show()
    print("Variables with Missing Qty : ", weather_data_missing.count().values)
    print("Total Missing values Qty : ", weather_data_missing.sum().values)

    f, ax = plt.subplots(5, 4, figsize=(15, 12))
    for idx, station in enumerate(sorted(weather_data_cp["station_nbr"].unique())):
        df = weather_data_cp.copy()
        station_weather = df[df["station_nbr"] == station]
        # Chck missing values
        weather_data_null = station_weather.isnull().sum()
        weather_data_null = weather_data_null.drop(weather_data_null[weather_data_null == 0].index).sort_values(
            ascending=False)
        weather_data_missing = pd.DataFrame({'Missing Numbers': weather_data_null})
        weather_data_null = weather_data_null / len(weather_data_cp) * 100
        # Barplot missing values
        sns.barplot(x=weather_data_null.index, y=weather_data_null \
                    , ax=ax[idx // 4, idx % 4])
        ax[idx // 4, 0].set_ylabel("Percentage")
        ax[idx // 4, idx % 4].set_ylim(0, 1)
        ax[idx // 4, idx % 4].set_xticks([])
        ax[idx // 4, idx % 4].set_xlabel("Station" + str(station))
    plt.suptitle('Missing values in each station in % \n Station #5 is almost made of missing values only')
    plt.show()

    # split year, month
    def make_year(date):
        return int(date.split('-')[0])

    def make_month(date):
        return int(date.split('-')[1])

    def tendency(store_number):
        '''
        input : store_nbr
        output : graph represetning total units for each year, each month
        '''
        df_tmp = sales_df.copy()
        df_tmp['year'] = df_tmp['date'].apply(make_year)
        df_tmp['month'] = df_tmp['date'].apply(make_month)
        # show yearly units tendency by store_nbr
        store_month = df_tmp.pivot_table(index=['year', 'month'], columns='store_nbr', values='units', aggfunc=np.sum)
        target_store = store_month[store_number]
        that_2012 = target_store.loc[2012]
        that_2013 = target_store.loc[2013]
        that_2014 = target_store.loc[2014]
        plt.figure(figsize=(12, 8))
        plt.plot(that_2012, label='2012', c='blue', ls='--', lw=4)
        plt.plot(that_2013, label='2013', c='green', ls=':', marker='D', ms=10, lw=4)
        plt.plot(that_2014, label='2014', c='red', lw=4)
        plt.legend(loc='best', prop={'size': 20})
        plt.show()


##### QUERIES #####

def items_sold_in_each_store(df_train):
    df_unit_sales = df_train.pivot_table(["units"], ["store_nbr"], ["item_nbr"], aggfunc=np.sum)
    print_df(df_unit_sales)


def find_most_sold_product(df_train):
    item_nbrs, store_nbrs = mf.report_item_sales(df_train)
    plt.figure(figsize=(15, 5))
    item_nbrs = np.array(item_nbrs)
    store_nbrs = np.array(store_nbrs)
    plt.xlabel("Item Number")
    plt.ylabel("Store #")
    plt.stem(item_nbrs, store_nbrs)
    ax = plt.subplot()
    idx = np.where(store_nbrs > 1)
    plt.scatter(item_nbrs[idx], store_nbrs[idx], s=150, c="r")
    item_nbrs_ = list(idx[0])
    loc = mf.setAnnotateLocation(item_nbrs_)
    utils.annotate_axes(range(len(item_nbrs_)), item_nbrs[item_nbrs_], \
                        list(zip(item_nbrs[item_nbrs_], store_nbrs[item_nbrs_])), loc, size="large", ax=ax)
    plt.suptitle('The most sold item is item #93')
    plt.show()


def number_of_units_sold_per_item(df_train):
    df_unit_sales = df_train.pivot_table(["units"], ["store_nbr"], ["item_nbr"], aggfunc=np.sum)
    units_sales = []

    for idx in list(df_unit_sales.units.columns):
        units_sales.append(df_unit_sales.units[idx].sum())
    units_sales = np.array(units_sales)

    plt.figure(figsize=(17, 6))

    plt.subplot(131)
    plt.xlabel("units")
    sns.distplot(units_sales)

    plt.subplot(132)
    stats.probplot(units_sales, plot=plt)

    plt.subplot(133)
    plt.xlabel("units")
    sns.boxplot(units_sales)
    plt.tight_layout()
    plt.show()


def number_of_units_sold_per_item_log_value(df_train):
    df_unit_sales = df_train.pivot_table(["units"], ["store_nbr"], ["item_nbr"], aggfunc=np.sum)
    units_sales = []

    for idx in list(df_unit_sales.units.columns):
        units_sales.append(df_unit_sales.units[idx].sum())
    units_sales = np.array(units_sales)

    nomalized_SalePrice = np.log1p(units_sales)

    plt.figure(figsize=(17, 6))

    plt.subplot(131)
    plt.xlabel("units")
    sns.distplot(nomalized_SalePrice)

    plt.subplot(132)
    stats.probplot(nomalized_SalePrice, plot=plt)

    plt.subplot(133)
    plt.xlabel("units")
    sns.boxplot(nomalized_SalePrice)
    plt.tight_layout()
    plt.show()


def valid_item_by_station(_df, is_get_mask=False, df_test=None):
    df = _df.copy()
    df['log1p'] = np.log1p(df['units'])

    store_and_item_group = df.groupby(["store_nbr", "item_nbr"])['log1p'].mean()
    store_and_item_group = store_and_item_group[store_and_item_group > 0.0]

    store_nbrs = store_and_item_group.index.get_level_values(0)
    item_nbrs = store_and_item_group.index.get_level_values(1)

    store_item_nbrs = sorted(zip(store_nbrs, item_nbrs), key=lambda t: t[1] * 10000 + t[0])

    if is_get_mask:
        li = [sno_ino for sno_ino in zip(df_test['store_nbr'], df_test['item_nbr'])]
        li_ = [sno_ino in store_item_nbrs for sno_ino in li]
        return li_

    li = [sno_ino for sno_ino in zip(df_train['store_nbr'], df_train['item_nbr'])]
    li_ = [sno_ino in store_item_nbrs for sno_ino in li]

    return df[li_].reset_index(drop=True)


def clean_data_station_5(weather_data_cp):
    df_weather_codesum = weather_data_cp.copy()
    df_weather_codesum = df_weather_codesum[df_weather_codesum["station_nbr"] != 5]
    return df_weather_codesum


def fix_weather_df(df_weather):
    df_weather['tmax'] = df_weather['tmax'].apply(mf.changeTypeToInt)
    df_weather['tmin'] = df_weather['tmin'].apply(mf.changeTypeToInt)
    df_weather['dewpoint'] = df_weather['dewpoint'].apply(mf.changeTypeToInt)
    df_weather['preciptotal'] = df_weather['preciptotal'].apply(mf.changeTypeToFloat)
    df_weather['stnpressure'] = df_weather['stnpressure'].apply(mf.changeTypeToFloat)
    df_weather['sealevel'] = df_weather['sealevel'].apply(mf.changeTypeToFloat)
    df_weather['resultspeed'] = df_weather['resultspeed'].apply(mf.changeTypeToFloat)
    df_weather['resultdir'] = df_weather['resultdir'].apply(mf.changeTypeToInt)
    df_weather['avgspeed'] = df_weather['avgspeed'].apply(mf.changeTypeToFloat)
    # interpolate()
    df_weather['tmax'] = df_weather['tmax'].interpolate()
    df_weather['tmin'] = df_weather['tmin'].interpolate()
    df_weather['dewpoint'] = df_weather['dewpoint'].interpolate()
    df_weather['preciptotal'] = df_weather['preciptotal'].interpolate()
    df_weather['stnpressure'] = df_weather['stnpressure'].interpolate()
    df_weather['resultspeed'] = df_weather['resultspeed'].interpolate()
    df_weather['resultdir'] = df_weather['resultdir'].interpolate()
    df_weather['avgspeed'] = df_weather['avgspeed'].interpolate()
    df_weather['tavg'] = df_weather['tavg'].apply(mf.changeTypeToInt)
    df_weather['tavg'] = (df_weather['tmax'] + df_weather['tmin']) / 2
    return df_weather


def show_tendency(store_number):
    '''
    input : store_nbr
    output : graph represetning total units for each year, each month
    '''

    # split year, month
    def make_year(date):
        return int(date.split('-')[0])

    def make_month(date):
        return int(date.split('-')[1])



    # show yearly units tendency by store_nbr
    store_month = sales_df.pivot_table(index=['year', 'month'], columns='store_nbr', values='units', aggfunc=np.sum)

    target_store = store_month[store_number]
    that_2012 = target_store.loc[2012]
    that_2013 = target_store.loc[2013]
    that_2014 = target_store.loc[2014]

    plt.figure(figsize=(12, 8))
    plt.plot(that_2012, label='2012', c='blue', ls='--', lw=4)
    plt.plot(that_2013, label='2013', c='green', ls=':', marker='D', ms=10, lw=4)
    plt.plot(that_2014, label='2014', c='red', lw=4)
    plt.legend(loc='best', prop={'size': 20})
    plt.show()


if __name__ == '__main__':
    """Import data"""
    weather_df = csv_to_df('DATA//weather.csv')
    key_df = csv_to_df('DATA//key.csv')
    sales_df = csv_to_df('DATA//sales.csv')
    test_df_weather = split_data_to_test(weather_df)
    train_df_weather = split_data_to_train(weather_df)
    test_df_sales = split_data_to_test(sales_df)
    train_df_sales = split_data_to_train(sales_df)


    """ PRINT DF FOR PROCESS VALIDATION """
    # print_df(train_df)
    # print_df(test_df)
    # print_df(sales_df)

    """{SECTION A} PLOT THE SALES DATA: 
        1. plot sales by date
        2. plot sales by store number
        3. plot sales by item number"""
    # plot_by_date(sales_df)
    # plot_by_store_numer(sales_df)
    # # plot_by_item_number(sales_df)
    # explore_data(train_df_sales)
    # explore_data_2(train_df_sales)
    # plot_missing_data(weather_df)

    """Find the stores contributed to each weather station"""
    """Each store has a different Meteorological Administration that receives weather information."""
    # print_df(mf.classifyStoresByStation(key_df).sort_values(by="station_nbr").reset_index(drop=True))
    # items_sold_in_each_store(train_df_sales)

    """FIND MOST SOLD ITEM"""
    # find_most_sold_product(train_df_sales)
    # number_of_units_sold_per_item(train_df_sales) # not so close to normal distribution
    # number_of_units_sold_per_item_log_value(train_df_sales)  # Scaling by log value --> closer to normal distribution

    """{SECTION B}DATA CLEANING"""
    """All rows for items that are not sold by each store have been removed"""
    # df_train = train_df_sales.copy()
    # df_train = valid_item_by_station(df_train)
    # print_df(df_train)
    # print(df_train.shape)

    """in station 5, more than 95% of 11 columns out of 19 are missing values,
   Since 4 columns have a missing value of 50% or more, remove them first"""
    # change the original df
    weather_df_Section_C = weather_df[weather_df["station_nbr"] != 5]
    # print_df(weather_df)


    # remove non informative features
    weather_df_Section_C = fix_weather_df(weather_df)
    # TODO drop those features as they are very empty
    weather_df_Section_C = weather_df.drop( columns= ['depart', 'sunrise', 'sunset', 'snowfall', 'codesum','sealevel'])


    for i in weather_df_Section_C.columns:
        weather_df_Section_C[i] = weather_df_Section_C[i].replace('-', 0)
        weather_df_Section_C[i] = weather_df_Section_C[i].replace(' ', 0)
        weather_df_Section_C[i] = weather_df_Section_C[i].replace('M', 0)
        weather_df_Section_C[i] = weather_df_Section_C[i].replace('T', 0)
        weather_df_Section_C[i] = weather_df_Section_C[i].replace('  T', 0)




    """MERGING THE TABLES USING THE CONNECTION TABLE key.csv"""
    # drop the item number and units
    df_store_number_and_date = sales_df[['date','store_nbr']]
    df_store_number_and_date = df_store_number_and_date.drop_duplicates(subset = ['date','store_nbr'])


    df_sales_inner_join_key = pd.merge(sales_df,key_df,on='store_nbr')
    # df_sales_inner_join_store_nbr_as_key = pd.merge(sales_df,key_df,on='store_nbr')
    # print('THIS IS df_sales_inner_join_key \n\n',df_sales_inner_join_key)

    df_weather_inner_join_key = pd.merge(weather_df_Section_C, key_df,on='station_nbr')
    # print('THIS IS df_weather_inner_join_key \n\n',df_weather_inner_join_key)

    # df_sales_inner_join_key = pd.merge(sales_df,key_df,on='store_nbr')
    merged_df = pd.merge(df_weather_inner_join_key,df_sales_inner_join_key, on=['store_nbr','station_nbr','date'])

    # #
    # print('THIS IS merged_df \n\n',merged_df)
    # show_tendency(2)

    # station_number_one_hot_encoded_with_weather = pd.get_dummies(df_sales_inner_join_weather, prefix='Store', columns=["store_nbr_y"],drop_first=False)
    # #TODO this is the table to use the model on

    """ Define a variable called KEY SUM """
    KEY_SUM = 0
    # Change

    "One-Hot encoding = For each store number (1-45) create a vector of 1/0 that represents the store"
    "replace the store number feature with the vector, creating 45 new features instead "
    # store_numbers_one_hot_encoded = pd.get_dummies(df_store_number_and_date, prefix='Store', columns=["store_nbr"],drop_first=False)
    # print(store_numbers_one_hot_encoded)

    store_nbr_list = []
    date_list = []
    KEY_SUM_list = []

    merged_df_only_with_key_sum_items = merged_df[((merged_df['item_nbr'] == 5) | (merged_df['item_nbr'] == 6) | (merged_df['item_nbr'] == 9) |
                                           (merged_df['item_nbr'] == 16) | (merged_df['item_nbr'] == 45))
                                                  & (merged_df['units'] != 0)]

    # print('THIS IS merged_df_only_with_key_sum_items \n',merged_df_only_with_key_sum_items)

    # a method to calculate the key sum by the given store and date
    def calculate_Key_sum(store_number,current_date):
        """EXTRACT ITEM NUMBER 5 """
        current_calc = merged_df_only_with_key_sum_items[(merged_df_only_with_key_sum_items['store_nbr'] == store_number) & (merged_df_only_with_key_sum_items['date'] == current_date)]

        sum_of_all_items = current_calc['units'].sum()
        return sum_of_all_items

    """ for each store 1-45, iter on each day (from 1/1/2012 until 31/10/2014)
        and calculate the Key sum (the sum of units 5,6,9,16,45).
        the Key_sum is the new feature that we'll add to the df, and later on would predict it using the weather data"""
    list_of_stores = []
    for s in df_store_number_and_date['store_nbr'].unique():
        list_of_stores.append(s)

    #TODO for each item, add as feature, and use his units as the "one-hot encoding"
    for s in list_of_stores:
        for d in df_store_number_and_date['date'].unique():
            store_nbr_list.append(s)
            date_list.append(d)
            KEY_SUM = calculate_Key_sum(s,d)
            KEY_SUM_list.append(KEY_SUM)




    # use the key sum values that was calculated into the final df
    data = {'store_nbr':store_nbr_list,'date':date_list,'key_sum':KEY_SUM_list}
    after_key_sum_df = pd.DataFrame(data)
    print_df(after_key_sum_df)

    final_df = pd.merge(after_key_sum_df, merged_df, on=['store_nbr', 'date'])
    # final_df = final_df.drop(columns=['item_nbr','station_nbr'])
    final_df = final_df.drop(columns=['item_nbr','station_nbr'])



    "One-Hot encoding :"
    "1. For each store number (1-45) create a vector of 1/0 that represents the store"
    "replace the store number feature with the vector, creating 45 new features instead"
    "2. For each item number (1-111) exclude the items 5,6,9,16,45, creating 107 new features instead"
    final_df = pd.get_dummies(final_df, prefix='Store', columns=["store_nbr"],drop_first=False)
    # final_df = pd.get_dummies(final_df, prefix='Item', columns=["item_nbr"],drop_first=False)
    # final_df = pd.get_dummies(final_df, prefix='Station_nbr', columns=["station_nbr"],drop_first=False)


    print('The final df to use ML models on:\n',final_df)
    final_df['date'] = pd.to_datetime(final_df['date'])
    # final_df = final_df.set_index('date')

    split_date = '2013-12-31'
    # df_training = final_df.loc[final_df['date'] <= split_date]
    # df_test = final_df.loc[final_df['date'] > split_date]

    #the features to build the model
    # train = final_df.loc[final_df['date'] <= split_date]
    train = final_df.iloc[:300000,:]
    # X_train = train.drop(columns=['key_sum','station_nbr'])

    train['year'] = train['date'].dt.year
    train['month'] = train['date'].dt.month
    train['day'] = train['date'].dt.day

    X_train = train.drop(columns=['key_sum','date'])
    y_train = train['key_sum']


    # test = final_df.loc[final_df['date'] > split_date]
    test =  final_df.iloc[300000:360000,:]
    # X_test = test.drop(columns=['key_sum','station_nbr'])

    test['year'] = test['date'].dt.year
    test['month'] = test['date'].dt.month
    test['day'] = test['date'].dt.day

    X_test = test.drop(columns=['key_sum','date'])
    y_test = test['key_sum']

    print('Train Dataset:\n', X_train)
    print('Test Dataset:\n', X_test)
    print('Train key sum vector:\n', y_train)
    print('Test key sum vector:\n', y_test)




    """{SECTION C} USE 2 MACHINE LEARNING MODELS TO PREDICT THE DAILY SALES FIGURES OF KEY_SUM"""

    ###### GRADIENT BOOSTING - Regressor#####
    # define model
    gb = GradientBoostingRegressor(random_state=10)

    # define parameter grid
    parameters_grid = {
        'learning_rate': [0.5, 1, 1.5],
        'n_estimators': [50, 100]
    }

    # define grid search
    grid_search = GridSearchCV(estimator=gb, param_grid=parameters_grid, cv=10)

    # fit estimator
    gb.fit(X_train, y_train)

    # # get best estimator
    # best = grid_search.best_estimator_
    #
    # # print best parameters
    # pd.DataFrame.from_dict(grid_search.best_params_, orient='index', columns=['Selected Value']).T
    # predict

    y_pred = gb.predict(X_test)

    # # calculate MSE
    # MSE = round(mean_squared_error(y_reg_test, y_pred), 3)
    # define function to calculate metrics
    # def get_metrics(y_test, y_pred):
    #
    #     actual_pos = y_test == 1
    #     actual_neg = y_test == 0
    #
    #     # get confusion matrix
    #     mat = metrics.confusion_matrix(y_test, y_pred)
    #     true_neg, false_pos, false_neg, true_pos = mat.ravel()
    #
    #     # calculate sensitivity and specificity
    #     sensitivity = round(true_pos / np.sum(actual_pos), 3)
    #     specificity = round(true_neg / np.sum(actual_neg), 3)
    #
    #     return sensitivity, specificity
    # acc = round(accuracy_score(y_test, y_pred), 3)
    # sensitivity, specificity = get_metrics(y_test, y_pred)
    # display metrics
    MSE = round(mean_squared_error(y_test, y_pred), 3)
    gb_df = pd.DataFrame([MSE]).T
    gb_df = gb_df.rename(index={0: 'Gradient boosting Regressor: '}, columns={0: 'MSE'})

    print(gb_df)

    ###### RANDOM FOREST - Regressor #########
    # define model
    rf = RandomForestClassifier(random_state=10)

    # define parameter grid
    parameters_grid = {
        'max_depth': [3, 4, 5],
        'min_samples_split': [2, 3, 4],
        'min_samples_leaf': [2, 4, 6],
        'n_estimators': [20, 50, 80]
    }

    # define model
    rf = RandomForestRegressor(n_jobs=-1,random_state=10)

    # define parameter grid
    parameters_grid = {
        'max_depth': [3, 5],
        'min_samples_leaf': [2, 8],
        'n_estimators': [20, 50],
        'max_features': [2, 4]
    }

    # define grid search
    grid_search = GridSearchCV(estimator=rf, param_grid=parameters_grid, cv=10)

    # fit estimator
    rf.fit(X_train, y_train)

    # # get best estimator
    # best = grid_search.best_estimator_
    #
    # # print best parameters
    # pd.DataFrame.from_dict(grid_search.best_params_, orient='index', columns=['Selected Value']).T

    # predict
    y_pred = rf.predict(X_test)

    # calculate MSE
    MSE = round(mean_squared_error(y_test, y_pred), 3)

    df_rf = pd.DataFrame([MSE]).T
    df_rf = df_rf.rename(index={0: 'Random Forest Regressor: '}, columns={0: 'MSE'})
    print(df_rf)




    """{SECTION D} USE 2 MACHINE LEARNING MODELS TO PREDICT IF ON A GIVEN DAY IT RAINED OR NOT FOR STORE NUMBER 11
    A rainy day is defined as a day in which the precipitation (preciptotal column), is greater than 0. Trace (T) is defined to be greater than 0."""


    # Trace (T) is defined to be greater than 0, so replace it with 0.1
    for i in weather_df.columns:
        weather_df[i] = weather_df[i].replace('-', 0)
        weather_df[i] = weather_df[i].replace(' ', 0)
        weather_df[i] = weather_df[i].replace('M', 0)
        weather_df[i] = weather_df[i].replace('T', 0.1)
        weather_df[i] = weather_df[i].replace('  T', 0.1)

    sales_without_store_11 = sales_df[sales_df['store_nbr'] != 11]
    # print('sales_without_store_11:\n',sales_without_store_11)

    df_weather_inner_join_station_nbr = pd.merge(weather_df, key_df,on='station_nbr')
    # print('THIS IS df_weather_inner_join_station_nbr \n\n',df_weather_inner_join_station_nbr)

    df_sales_inner_join_store_nbr = pd.merge(sales_without_store_11,key_df,on='store_nbr')
    # print('THIS IS df_sales_inner_join_store_nbr \n\n',df_sales_inner_join_store_nbr)

    df_merged_for_section_D = pd.merge(df_weather_inner_join_station_nbr,df_sales_inner_join_store_nbr, on=['store_nbr','station_nbr','date'])
    # print('THIS IS df_merged_for_section_D \n\n',df_merged_for_section_D)

    df_merged_for_section_D_after_filtering = df_merged_for_section_D[['date','store_nbr','item_nbr','units','preciptotal']]
    print('THIS IS df_merged_for_section_D_after_filtering \n\n',df_merged_for_section_D_after_filtering)
    df_merged_for_section_D_after_filtering = pd.get_dummies(df_merged_for_section_D_after_filtering, prefix='store', columns=["store_nbr"],drop_first=False)
    df_merged_for_section_D_after_filtering = pd.get_dummies(df_merged_for_section_D_after_filtering, prefix='Item', columns=["item_nbr"],drop_first=False)
    print('THIS IS after One hot on store and item: \n\n',df_merged_for_section_D_after_filtering)

    #TODO Pivot table -> converting each unit of an item into a column vector in the item number column

    # converted_item_and_units_into_row = pd.pivot_table(df_merged_for_section_D_after_filtering, values='units', index=['date','store_nbr'],
    #                 columns=['item_nbr'], aggfunc=np.sum, fill_value=0)
    # print('converted_item_and_units_into_row\n\n',converted_item_and_units_into_row)

    # df_merged_for_section_D_after_filtering['year'] = df_merged_for_section_D_after_filtering['date'].dt.year
    # df_merged_for_section_D_after_filtering['month'] = df_merged_for_section_D_after_filtering['date'].dt.month
    # df_merged_for_section_D_after_filtering['day'] = df_merged_for_section_D_after_filtering['date'].dt.day
    #
    # df_merged_for_section_D_after_filtering = df_merged_for_section_D_after_filtering.drop(columns=['date'])
    #
    # print('THIS IS after seperating date into year month and day: \n\n',df_merged_for_section_D_after_filtering)

    # convert preciptotal bigger than  0 to 1, and again every None (that was previously 0) again to 0
    df_merged_for_section_D_after_filtering.loc[df_merged_for_section_D_after_filtering['preciptotal'] > 0, 'preciptotal'] = 1
    df_merged_for_section_D_after_filtering.loc[df_merged_for_section_D_after_filtering['preciptotal'] == None, 'preciptotal'] = 0

    print('convert bigger than preciptotal 0 to 1: \n\n',df_merged_for_section_D_after_filtering)


    ################# TRAIN SECTION D ####################
    # X- the features to use. y- the class to predict
    split_date = '2013-12-31'

    train_Section_D = df_merged_for_section_D_after_filtering.loc[df_merged_for_section_D_after_filtering['date'] <= split_date]

    train_Section_D['year'] = train_Section_D['date'].dt.year
    train_Section_D['month'] = train_Section_D['date'].dt.month
    train_Section_D['day'] = train_Section_D['date'].dt.day

    train_Section_D = train_Section_D.drop(columns=['date'])

    # train_Section_D = df_merged_for_section_D_after_filtering.iloc[:1000000,:]
    X_train_Section_D = train_Section_D.drop(columns=['preciptotal'])
    y_train_Section_D = train_Section_D['preciptotal']


    ################# TEST SECTION D ####################
    test_Section_D = df_merged_for_section_D_after_filtering.loc[df_merged_for_section_D_after_filtering['date'] > split_date]

    test_Section_D['year'] = test_Section_D['date'].dt.year
    test_Section_D['month'] = test_Section_D['date'].dt.month
    test_Section_D['day'] = test_Section_D['date'].dt.day

    test_Section_D = test_Section_D.drop(columns=['date'])

    # test_Section_D =  df_merged_for_section_D_after_filtering.iloc[1000000:1200000,:]
    X_test_Section_D = test_Section_D.drop(columns=['preciptotal'])
    y_test_Section_D = test_Section_D['preciptotal']



    print('Train Dataset section D:\n', X_train_Section_D)
    print('Test Dataset section D:\n', X_test_Section_D)
    print('Train preciptotal vector:\n', y_train_Section_D)
    print('Test preciptotal vector:\n', y_test_Section_D)

    ######### AdaBoost - Classification #######
    # define model
    ab = AdaBoostClassifier(random_state=10)

    # # define parameter grid
    # parameters_grid = {
    #     'n_estimators': [20, 50]
    # }

    # # define grid search
    # grid_search = GridSearchCV(estimator=ab, param_grid=parameters_grid, cv=10)
    # fit estimator
    ab.fit(X_train_Section_D, y_train_Section_D)

    # # get best estimator
    # best = grid_search.best_estimator_
    # predict
    y_pred_Section_D = ab.predict(X_test_Section_D)
    # calculate accuracy
    acc = round(accuracy_score(y_test_Section_D, y_pred_Section_D), 3)

    df_D = pd.DataFrame([acc]).T
    df_D = df_D.rename(index={0: 'AdaBoost Classifier'}, columns={0: 'Accuracy'})
    print(df_D)

    ######### Random forest - Classification #######

    # define model
    rf_Section_D = RandomForestClassifier(random_state=10)

    # fit estimator
    rf_Section_D.fit(X_train_Section_D, y_train_Section_D)
    # predict
    y_pred_rf = rf_Section_D.predict(X_test_Section_D)
    # calculate accuracy
    acc = round(accuracy_score(y_test_Section_D, y_pred_Section_D), 3)

    rf_D = pd.DataFrame([acc]).T
    rf_D = rf_D.rename(index={0: 'Random Forest Classifier'}, columns={0: 'Accuracy'})
    print(rf_D)
    exit()