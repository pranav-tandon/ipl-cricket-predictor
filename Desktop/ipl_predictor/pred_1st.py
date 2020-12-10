import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib

def get_df(INNING_PATH):
    df = pd.read_csv(INNING_PATH)
    return df

def get_x(df):
    X = df.iloc[:, [1,2,3,4,5,6]]
    return X

def get_y(df, y_col):
    y = df.iloc[:, y_col]
    return y

def fit(X):
    return OneHotEncoder(categorical_features=[0, 1, 5]).fit(X)

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

def first_inn_gen(df, y_col, FILE_PATH_FIT, FILE_PATH_MODEL):
    X = get_x(df)
    y = get_y(df, y_col)
    X_fit = fit(X)
    dump_fit(FILE_PATH_FIT, X)
    X_train, X_test, y_train, y_test = get_train_test(X, y)
    model = get_model(X_train, y_train)
    dump_model(FILE_PATH_MODEL, model)
    return X_fit, model

def first_inn_pred(INNING_PATH, y_col, FILE_PATH_FIT, FILE_PATH_MODEL):
    df = get_df(INNING_PATH)
    first_inn_gen(df, y_col, FILE_PATH_FIT, FILE_PATH_MODEL)

DIR_PATH = os.path.dirname('__file__')
INNING_PATH = 'data/1st_inning.csv'
y_col = 7
FIT_PATH = 'data/pred/1st_inn/1st_inn_pred_fit.pkl'
MODEL_PATH = 'data/pred/1st_inn/1st_inn_pred.pkl'

first_inn_pred(INNING_PATH, y_col, FIT_PATH, MODEL_PATH)
