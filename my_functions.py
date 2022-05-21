from datetime import datetime
import pandas as pd
import json
import requests
import numpy as np
import scipy as sp
from dateutil.parser import parse


def isThereNoneData(df, percentage=60):

    columns = ["Column", "Row Count", "Missing Data", "M Data %", "Trace Data", "T Data %", "Bar Data", "B Data %"]
    li = list(df.columns)
    m_count_li = []
    t_count_li = []
    b_count_li = []
    row_count_li = []
    m_percent_li = []
    t_percent_li = []
    b_percent_li = []
    for l in li:
        m_count = 0
        t_count = 0
        b_count = 0
        for i, contents in enumerate(df[l]):
            contents = str(contents)
            if (contents.strip() == "M"):
                m_count += 1
            if (contents.strip() == "T"):
                t_count += 1
            if (contents.strip() == "-"):
                b_count += 1
        m_count_li.append(m_count)
        t_count_li.append(t_count)
        b_count_li.append(b_count)
        row_count_li.append(len(df[l]))
        m_percent_li.append(round(m_count / len(df[l]) * 100, 2))
        t_percent_li.append(round(t_count / len(df[l]) * 100, 2))
        b_percent_li.append(round(b_count / len(df[l]) * 100, 2))
    result_df = pd.DataFrame({
        "Column": li,
        "Row Count": row_count_li,
        "Missing Data": m_count_li,
        "M Data %": m_percent_li,
        "Trace Data": t_count_li,
        "T Data %": t_percent_li,
        "Bar Data": b_count_li,
        "B Data %": b_percent_li},
        columns=columns)

    for column in ["M Data %", "T Data %", "B Data %"]:
        print("{} over {}% : {}".format(" ".join(column.split(" ")[:-1]), \
                                        percentage, list(result_df["Column"][result_df[column] >= percentage])))
    return result_df


def changeTypeToInt(a):
    temp = str(a).strip()
    if temp == 'M' or temp == '-':
        return np.nan
    else:
        return int(a)


def changeTypeToFloat(a):
    temp = str(a).strip()
    if temp == 'M' or temp == '-':
        return np.nan
    elif temp == 'T':
        return 0
    else:
        return float(a)


def changeTypes(df, columns=[], print_msg=True):
    result_df = df.copy()
    if len(columns) == 0:
        columns = result_df.columns
    cols_int = ["tmax", "tmin", "tavg", "dewpoint", "wetbulb", "heat", "cool", "sunrise", "sunset"]
    cols_float = ["snowfall", "preciptotal", "stnpressure", "sealevel", "depart", "resultspeed", "resultdir",
                  "avgspeed"]

    process_int = []
    process_float = []
    for column in columns:
        if column in cols_int:
            process_int.append(column)
            result_df[column] = result_df[column].apply(changeTypeToInt)
        elif column in cols_float:
            process_float.append(column)
            result_df[column] = result_df[column].apply(changeTypeToFloat)



    return result_df


def remove_columns(df, columns=[], print_msg=True):

    for column in columns:
        df.drop(column, axis=1, inplace=True)

    if print_msg:
        print("제거한 컬럼명 : " + str(list(columns)))

    return df.tail()


def divideIntoNumericAndCategoricalVariables(df):

    numerical_features = []
    categorical_features = []
    for f in df.columns:
        if df.dtypes[f] != 'object' and f != "station_nbr":
            numerical_features.append(f)
        else:
            if f != "date":
                categorical_features.append(f)
    print("Numerical Features Qty :", len(numerical_features), "\n")
    print("Numerical Features : ", numerical_features, "\n\n")
    print("Categorical Features Qty :", len(categorical_features), "\n")
    print("Categorical Features :", categorical_features)
    return numerical_features, categorical_features


def getStoreList(df, item_nbr, msg=True):
    df_ = df.copy()
    df_ = df_[df_["item_nbr"] == item_nbr]
    df_ = df_[df_["units"] != 0]
    return len(list(df_["store_nbr"].unique()))


def report_item_sales(df):
    item_nbr_li = df["item_nbr"].unique()
    y_li = []
    for item_nbr in item_nbr_li:
        y_li.append(getStoreList(df, item_nbr))
    return item_nbr_li, y_li


def setAnnotateLocation(idx_li):
    result = []
    base_loc = (-8, 10)
    up_loc = (-8, 25)
    check_li = [22, 38, 83, 96]
    for i, idx in enumerate(idx_li):
        if idx in check_li:
            result.append(up_loc)
        else:
            result.append(base_loc)
        tmp = idx
    return result


def to_nan(a):
    temp = str(a).strip()
    if temp == 'M' or temp == '-':
        return np.nan
    elif temp == "T":
        return 0
    else:
        return float(a)


def nan_to_zero(a):
    if np.isnan(a):
        return 0
    else:
        return 1


def sum_missing_values(data):
    return len(data) - np.sum(data)


def returnWeatherMissingValueDataByStation(df_):
    df = df_.copy()
    weather_columns = ['tmax', 'tmin', 'tavg', 'depart', 'dewpoint', 'wetbulb', 'heat', 'cool', 'sunrise', 'sunset',
                       'snowfall', 'preciptotal', 'stnpressure', 'sealevel', 'resultspeed', 'resultdir', 'avgspeed']
    for column in weather_columns:
        df[column] = df[column].apply(to_nan)
        df[column] = df[column].apply(nan_to_zero)

    return df.pivot_table(weather_columns, "station_nbr", aggfunc=sum_missing_values)


def classifyStoresByStation(df):

    dictionary = {}
    for i, station_nbr in enumerate(df["station_nbr"]):
        store_nbr = str(df["store_nbr"].loc[i])
        if station_nbr in dictionary:
            dictionary[station_nbr] += ", " + store_nbr
        else:
            dictionary[station_nbr] = store_nbr

    return pd.DataFrame({"station_nbr": list(dictionary.keys()), "store_nbr": list(dictionary.values())})


def getCodesumUniqueDataList(codesum):
    codesums = str(codesum).strip().split(" ")
    for cd in codesums:
        if cd != "":
            if len(cd) > 3:
                if cd[:2] == format_d:
                    return 1
                if cd[2:] == format_d:
                    return 1
            else:
                if cd == format_d:
                    return 1
        else:
            return 0


def getCodesumUniqueDataList(df):
    df_codesum = df["codesum"]
    codesum_data_list = []
    for codesum in df_codesum:
        codesum = str(codesum).strip()
        cds = codesum.split(" ")
        if len(cds) != 0:
            for cd in cds:
                if cd != "":
                    if len(cd) > 3:
                        if cd[:2] not in codesum_data_list:
                            codesum_data_list.append(cd[:2])
                        if cd[2:] not in codesum_data_list:
                            codesum_data_list.append(cd[2:])
                    else:
                        if cd not in codesum_data_list:
                            codesum_data_list.append(cd)
    return codesum_data_list


def setCodesumUniqueDataList(codesum):
    codesums = str(codesum).strip().split(" ")

    for cd in codesums:
        if cd != "":
            if len(cd) > 3:
                if cd[:2] == format_d:
                    return 1
                if cd[2:] == format_d:
                    return 1
            else:
                if cd == format_d:
                    return 1
        else:
            return 0


def getCodesumFormats():
    codesum_formats = [
        "Rain", "Freezing", "Fog", "Mist", "Unknown Precipitation", \
        "HeavyFog", "Shallow", "Snow", "Haze", "ThunderStorm", "Vicinity", \
        "Drizzle", "Blowing", "Patches", "Widespread Dust", "Squall", "Ice Pellets", \
        "Smoke", "Hail", "Small Hail or Snow Pellets", "Snow Grains", "Partial", "Moderate"
    ]
    return codesum_formats


def getIndependentTwoSampleTest(df, codesum_list, df_moderate):
    result_li = []
    for codesum in codesum_list:
        df_ = df[df[codesum] != 0]
        day_length = len(df_)
        statistic, p_value = sp.stats.ttest_ind(df_["units"], df_moderate["units"], equal_var=False)
        if p_value <= 0.01:
            print(codesum_dict[codesum], codesum_num[codesum])
            print(round(p_value, 5))


def handle_date_Y(a):
    return parse(a).year


def handle_date_M(a):
    return parse(a).month


def handle_date_D(a):
    return parse(a).day


def isnan(a):
    if np.isnan(a):
        return 0
    else:
        return 1


def getSpecifiedWeatherData(df, station_nbr):

    return df[df["station_nbr"] == station_nbr].reset_index(drop=True)


def saveDataFrameToCsv(df, fileName, idx=False):
    fileName += "_" + datetime.now().strftime("%Y%m%d%H%M") + ".csv"
    return df.to_csv(fileName, index=idx)