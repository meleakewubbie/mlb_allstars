from mlb_data import (
    load_allstar_batters,
    load_allstar_pitchers,
    load_allstar_birth_state,
    load_batting_with_allstar_status,
    load_pitching_with_allstar_status
)
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


def players_avg_by_pos(players, exclude=None, include=None):
    '''
    Finds average stats for batters. positions can be optionally
    exluded or included by passing a list of those positions to the exclude
    or include parameter but not both
    '''
    assert(not (exclude and include))
    if exclude:
        mask = players['POS'].isin(exclude)
        players = players[~mask]

    elif include:
        mask = players['POS'].isin(include)
        players = players[mask]
    return players.groupby('POS').mean()


def fit_and_predict_allstars(X, y):
    '''
    Returns a decision tree regression model fitted to 80% split of the
    provided input (X) and (y) data. Also tests the model on the other
    20% of the data and informs the user of the model error.
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2,
                                                        random_state=0)
    regressor = DecisionTreeRegressor(random_state=0)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    err = mean_squared_error(y_test, y_pred)
    return regressor, err


def fit_allstars(X, y):
    '''
    Returns a decision tree regression model fitted to all of the provided
    input (X) and output (y) data without testing.
    '''
    regressor = DecisionTreeRegressor(random_state=0)
    regressor.fit(X, y)
    return regressor


def prepare_players_for_dtr(players):
    '''
    Prepares a provided set of players data (which should include an
    Is_Allstar column) for use with fit_and_predict_allstars() function.
    Prepares both the input (X) and output (y). Applies one-hot encoding
    to each accordiningly and removes irrelevant and encoded columns.
    Returns resulting input (X) and output (y) dataframes.
    '''
    X = players.copy()
    for pos in ['1B', '2B', '3B', 'C', 'CF', 'DH', 'LF', 'OF', 'RF', 'SS']:
        col = 'POS_' + pos
        X[col] = np.where(X['POS'] == pos, 1, 0)
    y = pd.DataFrame()
    y['0'] = np.where(players['Is_Allstar'] == 0, 1, 0)
    y['1'] = np.where(players['Is_Allstar'] == 1, 1, 0)
    X = X.drop(['playerID', 'yearID', 'POS', 'Is_Allstar'], axis=1)
    return X, y


def plot_allstars_by_state(out_file, year=None):
    '''
    Plotting all-stars by birth state (only considers USA) and
    saves the produced figure.
    '''
    bs = load_allstar_birth_state(year=year)
    bs = bs[bs['birthCountry'] == 'USA']
    bs_groups = bs.groupby('birthState').count()
    bs_groups.plot(kind='bar', y='playerID')
    plt.title('Allstar Players by Birth State')
    plt.xlabel('Birth State')
    plt.ylabel('Total')
    plt.savefig(out_file)


def model_allstar_batters():
    '''
    Generates a predictive model for allstar batters. Displays the
    error and returns the produced model.
    Allstar batters are chosen within year range of 1995-2015 and
    pitchers are not considered.
    '''
    bwa = load_batting_with_allstar_status(year=(1995, 2016),
                                           exclude_POS=['P'])
    X, y = prepare_players_for_dtr(bwa)
    model, err = fit_and_predict_allstars(X, y)
    print('bwa model error: {}'.format(err))
    return model


def predict_batters(model, players):
    y_pred = model.predict(players)
    return y_pred


def predict_nelson_cruz(model):
    '''
    Uses provided predictive model to predict whether or not Nelson Cruz,
    based on the data provided, is an all star.
    Prints the result of the prediction.
    '''
    test_df = pd.DataFrame({
        'playerID': ['nelson_cruz'],
        'yearID': [2018],
        'G': [144],
        'R': [70],
        'H': [133],
        'HR': [37],
        'RBI': [97],
        'SB': [1],
        'BB': [55],
        'POS': ['OF'],
        'Is_Allstar': [1]
    })
    X, y = prepare_players_for_dtr(test_df)
    print(X)
    y_pred = model.predict(X)
    print('Nelson Cruz AllStar Prediction:{}'.format(
        'allstar' if y_pred[0][1] == 1 else 'not allstar'))


def model_allstar_pitchers():
    '''
    Generates a predictive model for allstar pitchers. Displays the
    error and returns the produced model.
    Allstar pitchers are chosen within year range of 1995-2015 and
    where the number of games started (GS) is in range 15-100.
    '''
    pwa = load_pitching_with_allstar_status(year=(1995, 2016), GS=(15, 100))
    X, y = prepare_players_for_dtr(pwa)
    model, err = fit_and_predict_allstars(X, y)
    print('pwa model error: {}'.format(err))
    return model


def plot_avgs_by_pos(players, title, out_dir='plots'):
    '''
    Creates plots of total stats for each group of players, grouped by
    position. These plots are named based on the position being considered,
    the prefix title (title) provided, and the output direction (out_dir).
    '''
    df = players.drop(['playerID', 'yearID', 'startingPos'], axis=1)
    for pos in df['POS'].unique():
        data = df[df['POS'] == pos]
        data = data.drop(['POS'], axis=1)
        data = data.mean()
        data.plot(kind='bar')
        plt_title = title + '_' + pos
        plt.title(plt_title)
        plt.ylabel('total')
        plt.xlabel(pos + ' stats')
        plt.savefig(out_dir + '/' + plt_title + '.png')


def main():
    '''
    Main function where calculations are performed, results are generated,
    and data for answering questions is generated.
    '''
    # question 3
    plot_allstars_by_state('plots/1995-2015_birth_states.png',
                           year=(1995, 2016))
    plot_allstars_by_state('plots/1933-1995_birth_states.png',
                           year=(1933, 1996))
    # question 1
    allstar_batters = load_allstar_batters(year=(1995, 2016))
    plot_avgs_by_pos(allstar_batters, 'Avg. Allstar Batter Stats by Position')
    allstar_pitchers = load_allstar_pitchers(year=(1995, 2016))
    batters_avg_by_pos = players_avg_by_pos(allstar_batters, exclude=[])
    print(batters_avg_by_pos)
    batters_avg_by_pos.to_csv('output/batter_avg_by_pos.csv')
    pitchers_avg_by_pos = players_avg_by_pos(allstar_pitchers, include=['P'])
    print(pitchers_avg_by_pos)
    pitchers_avg_by_pos.to_csv('output/pitchers_avg_by_pos.csv')
    # question 2
    allstar_batters_model = model_allstar_batters()
    predict_nelson_cruz(allstar_batters_model)
    model_allstar_pitchers()


if __name__ == '__main__':
    main()
