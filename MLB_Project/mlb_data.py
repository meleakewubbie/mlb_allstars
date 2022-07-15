import pandas as pd
import numpy as np


def load_allstar_data():
    '''
    Loads dataset containing allstar players.
    '''
    allstar = pd.read_csv('data/AllstarFull.csv')
    allstar = allstar.astype({'playerID': 'string'})
    return allstar[['playerID', 'yearID', 'startingPos']]


def load_batting_data():
    '''
    Loads dataset containing allstar players.
    '''
    batting = pd.read_csv('data/Batting.csv')
    batting = batting.astype({'playerID': 'string'})
    return batting[['playerID', 'yearID', 'G', 'R', 'H', 'HR', 'RBI', 'SB',
                    'BB']]


def load_pitching_data():
    '''
    Returns dataframe containing allstar players.
    '''
    pitching = pd.read_csv('data/Pitching.csv')
    pitching = pitching.astype({'playerID': 'string'})
    return pitching[['playerID', 'yearID', 'W', 'L', 'GS', 'ERA', 'SO', 'R',
                     'IPouts']]


def load_fielding_data():
    '''
    Returns dataframe containing allstar players.
    '''
    fielding = pd.read_csv('data/Fielding.csv')
    fielding = fielding.astype({'playerID': 'string'})
    return fielding[['playerID', 'yearID', 'POS']]


def load_allstar_batters(year=None):
    '''
    Returns dataframe with allstar batters, their stats, and their positions.
    '''
    allstar = load_allstar_data()
    batting = load_batting_data()
    fielding = load_fielding_data()
    abf = pd.merge(allstar, batting, on=['playerID', 'yearID'],
                   suffixes=('', '_batting'))
    abf = pd.merge(abf, fielding, on=['playerID', 'yearID'],
                   suffixes=('', '_fielding'))
    # select rows within year range if parameter provided
    if year:
        start_year, end_year = year
        abf = abf[(abf['yearID'] >= start_year) & (abf['yearID'] < end_year)]
    return abf


def load_allstar_pitchers(year=None):
    '''
    Returns dataframe with allstar pitchers, their stats, and their positions.
    '''
    allstar = load_allstar_data()
    pitching = load_pitching_data()
    fielding = load_fielding_data()
    abf = pd.merge(allstar, pitching, on=['playerID', 'yearID'],
                   suffixes=('', '_pitching'))
    abf = pd.merge(abf, fielding, on=['playerID', 'yearID'],
                   suffixes=('', '_fielding'))
    if year:
        start_year, end_year = year
        abf = abf[(abf['yearID'] >= start_year) & (abf['yearID'] < end_year)]
    return abf


def load_batting_with_allstar_status(year=None, exclude_POS=None,
                                     include_POS=None):
    '''
    Returns dataframe containing batters data, combined with fielding data,
    and a column 'Is_Allstar' which indicates whether or not they are an
    allstar.
    '''
    allstar = load_allstar_data()
    batting = load_batting_data()
    fielding = load_fielding_data()
    abf = pd.merge(batting, fielding, on=['playerID', 'yearID'],
                   suffixes=('', '_batting'))
    # select rows within year range if parameter provided
    if year:
        start_year, end_year = year
        abf = abf[(abf['yearID'] >= start_year) & (abf['yearID'] < end_year)]
    assert(not (exclude_POS and include_POS))
    if exclude_POS:
        mask = abf['POS'].isin(exclude_POS)
        abf = abf[~mask]
    elif include_POS:
        mask = abf['POS'].isin(include_POS)
        abf = abf[mask]
    abf['Is_Allstar'] = np.where(abf['playerID'].isin(allstar['playerID']),
                                 1, 0)
    return abf


def load_pitching_with_allstar_status(year=None, GS=None):
    '''
    Returns dataframe containing pitchers data, combined with fielding data,
    and a column 'Is_Allstar' which indicates whether or not they are an
    allstar.
    '''
    allstar = load_allstar_data()
    pitching = load_pitching_data()
    fielding = load_fielding_data()
    apf = pd.merge(pitching, fielding, on=['playerID', 'yearID'],
                   suffixes=('', '_pitching'))
    # select rows within year range if parameter provided
    if year:
        start_year, end_year = year
        apf = apf[(apf['yearID'] >= start_year) & (apf['yearID'] < end_year)]
    # select rows within GS range if parameter provided
    if GS:
        start_GS, end_GS = GS
        apf = apf[(apf['GS'] >= start_GS) & (apf['GS'] < end_GS)]

    apf['Is_Allstar'] = np.where(apf['playerID'].isin(allstar['playerID']),
                                 1, 0)
    return apf


def select_by_year_range(df, year, column='yearID'):
    '''
    Returns rows from dataframe where 'year' column is within year range.
    '''
    start_year, end_year = year
    new_df = df[(df[column] >= start_year) & (df[column] < end_year)]
    return new_df


def load_allstar_birth_state(year=None):
    '''
    Loads dataset containing personal information of allstars.
    '''
    allstar = load_allstar_data()
    player_info = load_master_data()
    allstar_player_info = pd.merge(allstar, player_info, on=['playerID'],
                                   suffixes=('', '_info'))
    if year:
        allstar_player_info = select_by_year_range(allstar_player_info, year)
    return allstar_player_info


def load_master_data():
    '''
    Loads dataset containing personal information of players.
    '''
    player_info = pd.read_csv('data/Master.csv')
    player_info = player_info.astype({'playerID': 'string'})
    return player_info
