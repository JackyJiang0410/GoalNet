import warnings
import pandas as pd
from tqdm import tqdm
from socceraction.data.statsbomb import StatsBombLoader
import socceraction.spadl as spadl
import numpy as np
import socceraction.xthreat as xthreat

def get_event_data(competition_name, season_name):
    warnings.filterwarnings(action="ignore", message="credentials were not supplied. open data access only")
    SBL = StatsBombLoader(getter="remote", creds={"user": None, "passwd": None})

    # Load competitions and games
    competitions = SBL.competitions()
    selected_competitions = competitions[
        (competitions.competition_name == competition_name) &
        (competitions.season_name == season_name)
    ]
    games = pd.concat([
        SBL.games(row.competition_id, row.season_id)
        for row in selected_competitions.itertuples()
    ])

    # Display games information
    games[["home_team_id", "away_team_id", "game_date", "home_score", "away_score"]]

    # Load game data
    games_verbose = tqdm(list(games.itertuples()), desc="Loading game data")
    teams, players, actions_list = [], [], []
    for game in games_verbose:
        teams.append(SBL.teams(game.game_id))
        players.append(SBL.players(game.game_id))
        events = SBL.events(game.game_id)
        actions_df = spadl.statsbomb.convert_to_actions(
            events,
            home_team_id=game.home_team_id,
            xy_fidelity_version=1,
            shot_fidelity_version=1
        )
        actions_df['game_id'] = game.game_id
        actions_list.append(actions_df)

    # Concatenate data
    actions = pd.concat(actions_list, ignore_index=True)
    teams = pd.concat(teams).drop_duplicates(subset="team_id")
    players = pd.concat(players).drop_duplicates(subset="player_id")
    actions = spadl.add_names(actions)

    # Load events and calculate possession change and recipient information
    event = []
    for game in games_verbose:
        events = SBL.events(game.game_id)
        event.append(events)
    event = pd.concat(event)
    event['possession_change'] = event['possession_team_id'] != event['team_id']
    event['pass_dict'] = event['extra'].apply(lambda x: x['pass'] if 'pass' in x.keys() else {})
    event['recipient_id'] = event['pass_dict'].apply(lambda x: int(x['recipient']['id']) if 'recipient' in x.keys() else None)

    possess_reci = event[['possession_change', 'recipient_id', 'event_id']]

    # Load and fit the Expected Threat (xT) model
    url_grid = "https://karun.in/blog/data/open_xt_12x8_v1.json"
    xTModel = xthreat.load_model(url_grid)
    xTModel = xthreat.ExpectedThreat(l=16, w=12)
    xTModel.fit(actions)

    # Calculate xT values
    actions["xT_value"] = xTModel.rate(actions)

    # Merge actions with possession change and recipient information
    merged_actions = pd.merge(
        actions, possess_reci, left_on='original_event_id', right_on='event_id', how='left'
    ).drop(columns=['event_id'])

    # Add player names and receiver names
    merged_actions = pd.merge(
        merged_actions, players[['player_id', 'player_name']], left_on='player_id', right_on='player_id', how='left'
    )
    merged_actions = pd.merge(
        merged_actions, players[['player_id', 'player_name']], left_on='recipient_id', right_on='player_id', how='left', suffixes=('', '_recipient')
    ).drop(columns=['player_id_recipient'])

    # Calculate xT change
    merged_actions['xT_value'] = merged_actions['xT_value'].fillna(0)
    merged_actions['prev_xT'] = merged_actions['xT_value'].shift(1)
    merged_actions['prev_game_id'] = merged_actions['game_id'].shift(1)
    merged_actions['xT_change'] = np.select(
        condlist=[
            merged_actions['game_id'] != merged_actions['prev_game_id'],
            merged_actions['possession_change'] == True,
            merged_actions['possession_change'] == False,
        ],
        choicelist=[
            0,
            merged_actions['xT_value'] + merged_actions['prev_xT'],
            merged_actions['xT_value'] - merged_actions['prev_xT'],
        ],
        default=0
    )

    actions_final = merged_actions[['game_id', 'original_event_id', 'period_id', 'time_seconds', 'team_id', 'player_id',
                                    'player_name', 'start_x', 'start_y', 'end_x', 'end_y', 'type_id', 'result_id',
                                    'bodypart_id', 'action_id', 'xT_value', 'xT_change', 'player_name_recipient']]
    
    return actions_final


