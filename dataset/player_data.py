import pandas as pd

def get_player_data(filepath):
    players = pd.read_csv(filepath)
    players = players.drop(columns=['Unnamed: 0', 'assists', 'bigChancesMissed', 'totalShots', 'errorLeadToGoal'])
    players = players.set_index('Player')

    return players