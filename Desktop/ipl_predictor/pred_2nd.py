import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib


def get_df(INNING_PATH):
    df = pd.read_csv(INNING_PATH)
    df['winner'] = np.where(df['winner'] == df['batting_team'], 1, 0)
    return df

def get_bat_df(df):
    return df[(df["winner"] == 1)]

def get_bowl_df(df):
    return df[(df["winner"] == 0)]

def get_x(df):
    X = df.iloc[:, [2, 3, 4, 5, 6, 10, 11]]
    return X

def get_y(df, y_col):
    y = df.iloc[:, y_col]
    return y

def fit(X):
    return OneHotEncoder(categorical_features=[0, 1, 6]).fit(X)

def get_train_test(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    return X_train, X_test, y_train, y_test

def get_model(X, y):
    return RandomForestRegressor(n_estimators=10, random_state=0).fit(X, y)

def dump_fit(FILE_PATH, X):
    FINAL_PATH = os.path.join(DIR_PATH, FILE_PATH)
    X_fit = fit(X)
    joblib.dump(X_fit, FINAL_PATH)

def dump_model(FILE_PATH, model):
    FINAL_PATH = os.path.join(DIR_PATH, FILE_PATH)
    joblib.dump(model, FINAL_PATH)

def second_inn_gen(df, y_col, FILE_PATH_FIT, FILE_PATH_MODEL):
    X = get_x(df)
    y = get_y(df, y_col)
    X_fit = fit(X)
    dump_fit(FILE_PATH_FIT, X)
    X_train, X_test, y_train, y_test = get_train_test(X, y)
    model = get_model(X_train, y_train)
    dump_model(FILE_PATH_MODEL, model)
    return X_fit, model

def second_inn_win_end(INNING_PATH, y_col, FILE_PATH_FIT, FILE_PATH_MODEL):
    df = get_df(INNING_PATH)
    second_inn_gen(df, y_col, FILE_PATH_FIT, FILE_PATH_MODEL)

def second_inn_bat(INNING_PATH, y_col, FILE_PATH_FIT, FILE_PATH_MODEL):
    df = get_df(INNING_PATH)
    X_bat = get_bat_df(df)
    second_inn_gen(X_bat, y_col, FILE_PATH_FIT, FILE_PATH_MODEL)

def second_inn_bowl(INNING_PATH, y_col, FILE_PATH_FIT, FILE_PATH_MODEL):
    df = get_df(INNING_PATH)
    X_bowl = get_bowl_df(df)
    second_inn_gen(X_bowl, y_col, FILE_PATH_FIT, FILE_PATH_MODEL)

DIR_PATH = os.path.dirname('__file__')
INNING_PATH = 'data/2nd_inning.csv'
y_win_col = 7
y_end_col = 13
y_bat_col = 9
y_bowl_col = 8
WIN_FIT_PATH = 'data/pred/2nd_inn/who_win/2nd_inn_win_pred_fit.pkl'
END_FIT_PATH = 'data/pred/2nd_inn/end/2nd_inn_end_fit.pkl'
BAT_FIT_PATH = 'data/pred/2nd_inn/bat_win/2nd_inn_bat_win_wicket_fit.pkl'
BOWL_FIT_PATH = 'data/pred/2nd_inn/bowl_win/2nd_inn_bowl_win_run_fit.pkl'
WIN_MODEL_PATH = 'data/pred/2nd_inn/who_win/2nd_inn_win_pred.pkl'
END_MODEL_PATH = 'data/pred/2nd_inn/end/2nd_inn_end.pkl'
BAT_MODEL_PATH = 'data/pred/2nd_inn/bat_win/2nd_inn_bat_win_wicket.pkl'
BOWL_MODEL_PATH = 'data/pred/2nd_inn/bowl_win/2nd_inn_bowl_win_run.pkl'

second_inn_win_end(INNING_PATH, y_win_col, WIN_FIT_PATH, WIN_MODEL_PATH)
second_inn_win_end(INNING_PATH, y_end_col, END_FIT_PATH, END_MODEL_PATH)
second_inn_bat(INNING_PATH, y_bat_col, BAT_FIT_PATH, BAT_MODEL_PATH)
second_inn_bowl(INNING_PATH, y_bowl_col, BOWL_FIT_PATH, BOWL_MODEL_PATH)