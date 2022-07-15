import pandas as pd
import mlb_data


def test_select_by_year_range():
    '''
    Test the mlb_data.select_by_year_range() function on an example
    dataframe.
    '''
    df = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'year': [2000, 2003, 2005, 2007, 2009]
    })
    expected_result = pd.DataFrame({
        'id': [3, 4],
        'year': [2005, 2007]
    })
    result = mlb_data.select_by_year_range(df, (2004, 2009), column='year')
    result = result.reset_index(drop=True)
    assert(expected_result.equals(result))


def main():
    '''
    Run all tests for mlb_data_test.py
    '''
    test_select_by_year_range()


if __name__ == '__main__':
    main()
