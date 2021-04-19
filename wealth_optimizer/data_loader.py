import os
from typing import List
import datetime as dt
import numpy as np
import pandas as pd

import investpy
from wealth_optimizer.constants import SECTOR_ETFS

__all__ = ['ExcelDataReader', 'load_csv_data', 'load_sector_etfs']


class ExcelDataReader:
    """
    Read returns data from xlsx file.
    The column names must start with either "asset_" or "feature_". Asset columns are returns of a traded asset.
    Feature columns are indicators (economic or other data) that is available at the time.
    """

    def __init__(self, path_to_file):

        if not isinstance(path_to_file, str):
            raise ValueError('path_to_file must be a string')

        if not os.path.exists(path_to_file):
            raise FileNotFoundError('cannot find: {}'.format(path_to_file))

        if not path_to_file.endswith('.xlsx'):
            raise ValueError('not a valid excel file: {}'.format(path_to_file))

        # load data
        returns_data = pd.read_excel(path_to_file, index_col=0)
        self.validate_data(returns_data)
        self.assets = [x for x in list(returns_data.columns) if 'feature_' not in x]
        self.features = [x for x in list(returns_data.columns) if 'feature_' in x]

        if len(self.assets) < 2:
            raise ValueError('dataset must have at least two tradeable assets')

        self.data = returns_data

    @staticmethod
    def validate_data(data: pd.DataFrame):

        for col in data.columns:
            if 'feature_' not in col:
                if 'asset_' not in col:
                    if 'benchmark_' not in col:
                        raise ValueError('Data validation failed, invalid column: {}'.format(col))


def load_csv_data(file_path, assets, save_pickle=False, save_filename='dataset.pkl', add_cash_asset=False):
    price_data = pd.read_csv(file_path, index_col=0)

    # compute daily returns
    return_columns = []
    for asset in assets:
        col_name = asset + '_Return'
        return_columns.append(col_name)
        price_data[col_name] = (price_data[asset + '_Close'] - price_data[asset + '_Open']) / price_data[
            asset + '_Open']

    returns_data = price_data[return_columns].copy()
    if add_cash_asset:
        returns_data['risk_free_asset'] = np.zeros(len(price_data))

    if save_pickle:
        returns_data.to_pickle(save_filename)
    return returns_data


def load_sector_etfs(etfs: List, start_date: dt.date, end_date: dt.date, frequency: str = 'Daily',
                     save_file: str = 'sectors.pkl') -> pd.DataFrame:
    """
    Load data for a list of etfs
    """
    if start_date > end_date:
        raise ValueError('start date cannot be later than end date')

    if frequency not in ['Daily', 'Weekly', 'Monthly']:
        raise ValueError('invalid frequency')

    for etf in etfs:
        assert etf in SECTOR_ETFS.keys(), 'etf not found: {}'.format(etf)

    start_date_str = start_date.strftime('%d/%m/%Y')
    end_date_str = end_date.strftime('%d/%m/%Y')
    country = 'united states'

    output_df = pd.DataFrame(columns=etfs)

    for etf in etfs:
        etf_string = SECTOR_ETFS[etf]
        d = investpy.get_etf_historical_data(etf_string, country, start_date_str, end_date_str, interval=frequency)
        output_df[etf] = (d['Close'] - d['Open']) / d['Open']

    return output_df
