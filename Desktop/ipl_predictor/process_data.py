import pandas as pd
import numpy as np

def get_match(id):
    return match_data[(match_data["id"] == id)]

def get_winner(match):
     match_info = get_match(match).values
     winner =  match_info[0,10]
     by_run =  match_info[0,11]
     by_wicket = match_info[0,12]
     return (winner,by_run,by_wicket)
 
def get_inning_match(inning):
    data = delivery_data[(delivery_data["inning"]==inning)]
    inn = delivery_data[(delivery_data["inning"]==1)]
    return_data = []
    match_range = 2000
    for match in range(1,match_range):
        match_info = get_match(match)
        winner =  get_winner(match)
        for_run = inn[(inn["match_id"]==match)]
        for_run["run"] = for_run.total_runs.cumsum()
        run = for_run['run'].max()
        match = data[(data["match_id"]==match)]
        match["run"] = match.total_runs.cumsum()
        match["balls"] = 6 * (match["over"] - 1 ) + match["ball"]
        match["player_dismissed"] = np.where(match["player_dismissed"].isnull(),0,1)
        match["wicket"] = match.player_dismissed.cumsum()
        match["winner"] = winner[0]
        match["win_by_runs"] =  winner[1]
        match["win_by_wickets"] =  winner[2]
        match["1st_inning_run"] = run
        
        city = match_info["city"].astype(str)
        city_id = np.nan 
        for index, row in city_data.iterrows():
            if row['name'] in str(city):
                city_id = row['id']
                break
        match["city"] = city_id
        return_data.append(match)

    my_data = pd.DataFrame(data)
    return my_data

def get_1st_inning_total_run():
    data = delivery_data[(delivery_data["inning"]==2)]
    return_data = []
    match_range = 2000
    for match in range(1, match_range):
        match = data[(data["match_id"]==match)]
        match["run"] = match.total_runs.cumsum()
        match["total"] = match['run'].max()
        return_data.append(match)
    data = pd.concat(return_data).iloc[:, [22]].values
    data = pd.DataFrame(data)
    return data

def add_city_code():
    return_data = []
    match_range = 2000
    for match in range(1, match_range):
        match = get_match(match)

        city = match["city"].astype(str)
        city_id = np.nan
        for index, row in city_data.iterrows():
            if row['name'] in str(city):
                city_id = row['id']
                break
        match["city_id"] = city_id
        return_data.append(match)
    my_data = pd.DataFrame(data)
    return my_data

delivery_data = pd.read_csv('data/deliveries.csv')
match_data = pd.read_csv('data/matches.csv')
team_data = pd.read_csv('data/teams.csv')
city_data = pd.read_csv('data/city_id.csv')

second_inn = pd.DataFrame(get_inning_match(2).join(get_1st_inning_total_run()).iloc[:,[0,2,3,21,22,23,24,25,26,27,28,29]].values)
second_inn.to_csv("data/2nd_inning.csv")
add_city_code().to_csv('data/match_with_city.csv')

