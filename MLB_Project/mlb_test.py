import pandas as pd
import mlb


def test_prepare_players_for_dtr():
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
    X_expected = pd.DataFrame({
        'G': [144],
        'R': [70],
        'H': [133],
        'HR': [37],
        'RBI': [97],
        'SB': [1],
        'BB': [55],
        'POS_1B': [0],
        'POS_2B': [0],
        'POS_3B': [0],
        'POS_C': [0],
        'POS_CF': [0],
        'POS_DH': [0],
        'POS_LF': [0],
        'POS_OF': [1],
        'POS_RF': [0],
        'POS_SS': [0],
    })
    y_expected = pd.DataFrame({
        '0': [0],
        '1': [1]
    })
    X, y = mlb.prepare_players_for_dtr(test_df)
    assert(X.equals(X_expected))
    assert(y.equals(y_expected))


def main():
    test_prepare_players_for_dtr()


if __name__ == '__main__':
    main()
