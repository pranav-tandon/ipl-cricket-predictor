from sklearn.externals import joblib
import os
import numpy as np

DIR_PATH = os.path.dirname(__file__)

def pred_gen(X_test, ABS_FIT_PATH, ABS_MODEL_PATH):
    model = joblib.load(ABS_MODEL_PATH)
    X_test = (joblib.load(ABS_FIT_PATH)).transform(X_test).toarray()
    y_pred = model.predict(X_test)
    return y_pred[0]

def pre_match_predict(season,team1,team2,city):
    X_test = [[season,team1,team2,city]]
    FIT_PATH = "pre_pred/pre_pred_fit.pkl"
    REG_PATH = "pre_pred/pre_pred.pkl"
    ABS_FIT_PATH = os.path.join(DIR_PATH, FIT_PATH)
    ABS_MODEL_PATH = os.path.join(DIR_PATH, REG_PATH)
    y_pred = pred_gen(X_test, ABS_FIT_PATH, ABS_MODEL_PATH)
    return y_pred

def predict_1st_inn(team_batting,team_bowling,run,ball,wicket,city):
    X_test = [[team_batting,team_bowling,run,ball,wicket,city]]
    FIT_PATH = "1st_inn/1st_inn_pred_fit.pkl"
    REG_PATH = "1st_inn/1st_inn_pred.pkl"
    ABS_FIT_PATH = os.path.join(DIR_PATH, FIT_PATH)
    ABS_MODEL_PATH = os.path.join(DIR_PATH, REG_PATH)
    y_pred = pred_gen(X_test, ABS_FIT_PATH, ABS_MODEL_PATH)
    return y_pred

def predict_if_bat_win(team_batting,team_bowling,run,ball,wicket,target,city):
    X_test = [[team_batting,team_bowling,run,ball,wicket,target,city]]
    FIT_PATH = "2nd_inn/bat_win/2nd_inn_bat_win_wicket_fit.pkl"
    REG_PATH = "2nd_inn/bat_win/2nd_inn_bat_win_wicket.pkl"
    ABS_FIT_PATH = os.path.join(DIR_PATH, FIT_PATH)
    ABS_MODEL_PATH = os.path.join(DIR_PATH, REG_PATH)
    y_pred = pred_gen(X_test, ABS_FIT_PATH, ABS_MODEL_PATH)
    return y_pred

def predict_if_bowl_win(team_batting,team_bowling,run,ball,wicket,target,city):
    X_test = [[team_batting,team_bowling,run,ball,wicket,target,city]]
    FIT_PATH = "2nd_inn/bowl_win/2nd_inn_bowl_win_run_fit.pkl"
    REG_PATH = "2nd_inn/bowl_win/2nd_inn_bowl_win_run.pkl"
    ABS_FIT_PATH = os.path.join(DIR_PATH, FIT_PATH)
    ABS_MODEL_PATH = os.path.join(DIR_PATH, REG_PATH)
    y_pred = pred_gen(X_test, ABS_FIT_PATH, ABS_MODEL_PATH)
    return y_pred

def predict_2nd_end_ball(team_batting,team_bowling,run,ball,wicket,target,city):
    X_test = [[team_batting,team_bowling,run,ball,wicket,target,city]]
    FIT_PATH = "2nd_inn/end/2nd_inn_end_fit.pkl"
    REG_PATH = "2nd_inn/end/2nd_inn_end.pkl"
    ABS_FIT_PATH = os.path.join(DIR_PATH, FIT_PATH)
    ABS_MODEL_PATH = os.path.join(DIR_PATH, REG_PATH)
    y_pred = pred_gen(X_test, ABS_FIT_PATH, ABS_MODEL_PATH)
    return y_pred

def predict_2nd_inn(team_batting,team_bowling,run,ball,wicket,target,city):
    X_test = [[team_batting,team_bowling,run,ball,wicket,target,city]]
    FIT_PATH = "2nd_inn/who_win/2nd_inn_win_pred_fit.pkl"
    REG_PATH = "2nd_inn/who_win/2nd_inn_win_pred.pkl"
    ABS_FIT_PATH = os.path.join(DIR_PATH, FIT_PATH)
    ABS_MODEL_PATH = os.path.join(DIR_PATH, REG_PATH)
    y_pred = pred_gen(X_test, ABS_FIT_PATH, ABS_MODEL_PATH)
    end_ball = predict_2nd_end_ball(team_batting,team_bowling,run,ball,wicket,target,city)
    if y_pred<0.5:info = predict_if_bowl_win(team_batting,team_bowling,run,ball,wicket,target,city)
    else: info = predict_if_bat_win(team_batting,team_bowling,run,ball,wicket,target,city)
    return [y_pred,info,end_ball]

def get_team(id):
    teams = {
            "1":'Sunrisers Hyderabad',
            "2":'Royal Challengers Bangalore',
            "3":'Chennai Super Kings',
            "4":'Kings XI Punjab',
            "5":'Rajasthan Royals',
            "6":'Delhi Capitals',
            "7":'Mumbai Indians',
            "8":'Kolkata Knight Riders'
    }
    return teams[id]
