import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from mpl_finance import candlestick2_ohlc
from sklearn.cluster import AgglomerativeClustering

class InvalidParameterException(Exception):
    pass

class InvalidArgumentException(Exception):
    pass

class BaseScorer:

    def fit(self, levels, ohlc_df):
        raise NotImplementedError()

class BaseLevelFinder:

    def __init__(self, merge_distance, merge_percent=None, level_selector='median'):

        self._merge_distance = merge_distance
        self._merge_percent = merge_percent

        self._level_selector = level_selector

        self._levels = None
        self._validate_init_args()

    @property
    def levels(self):
        return self._levels

    def _validate_init_args(self):
        pass

    def fit(self, data):
        if isinstance(data, pd.DataFrame):
            X = data['Close'].values
        elif isinstance(data, np.array):
            X = data
        else:
            raise InvalidArgumentException(
                'Only np.array and pd.DataFrame are supported in `fit` method'
            )

        prices = self._find_potential_level_prices(X)
        levels = self._aggregate_prices_to_levels(prices, self._get_distance(X))

        self._levels = levels

    def _find_potential_level_prices(self, X):
        raise NotImplementedError()

    def _get_distance(self, X):
        if self._merge_distance:
            return self._merge_distance

        mean_price = np.mean(X)
        return self._merge_percent * mean_price / 100

    def _aggregate_prices_to_levels(self, pivot_prices, distance):
        raise NotImplementedError()


class BaseZigZagLevels(BaseLevelFinder):

    def __init__(self, peak_percent_delta, merge_distance, merge_percent=None, min_bars_between_peaks=0, peaks='All',
                 level_selector='median'):
        self._peak_percent_delta = peak_percent_delta / 100
        self._min_bars_between_peaks = min_bars_between_peaks
        self._peaks = peaks
        super().__init__(merge_distance, merge_percent, level_selector)

    def _find_potential_level_prices(self, X):
        pivots = peak_valley_pivots(X, self._peak_percent_delta, -self._peak_percent_delta)
        indexes = self._get_pivot_indexes(pivots)
        pivot_prices = X[indexes]

        return pivot_prices

    def _get_pivot_indexes(self, pivots):
        if self._peaks == 'All':
            indexes = np.where(np.abs(pivots) == 1)
        elif self._peaks == 'High':
            indexes = np.where(pivots == 1)
        elif self._peaks == 'Low':
            indexes = np.where(pivots == -1)
        else:
            raise InvalidParameterException(
                'Peaks argument should be one of: `All`, `High`, `Low`'
            )

        return indexes if self._min_bars_between_peaks == 0 else self._filter_by_bars_between(indexes)

    def _filter_by_bars_between(self, indexes):
        indexes = np.sort(indexes).reshape(-1, 1)

        try:
            selected = [indexes[0][0]]
        except IndexError:
            return indexes

        pre_idx = indexes[0][0]
        for i in range(1, len(indexes)):
            if indexes[i][0] - pre_idx < self._min_bars_between_peaks:
                continue
            pre_idx = indexes[i][0]
            selected.append(pre_idx)

        return np.array(selected)


def _cluster_prices_to_levels(prices, distance, level_selector='mean'):
    clustering = AgglomerativeClustering(distance_threshold=distance, n_clusters=None)
    try:
        clustering.fit(prices.reshape(-1, 1))
    except ValueError:
        return None

    df = pd.DataFrame(data=prices, columns=('price',))
    df['cluster'] = clustering.labels_
    df['peak_count'] = 1

    grouped = df.groupby('cluster').agg(
        {
            'price': level_selector,
            'peak_count': 'sum'
        }
    ).reset_index()

    return grouped.to_dict('records')


class ZigZagClusterLevels(BaseZigZagLevels):

    def _aggregate_prices_to_levels(self, prices, distance):
        return _cluster_prices_to_levels(prices, distance, self._level_selector)


class RawPriceClusterLevels(BaseLevelFinder):
    def __init__(self, merge_distance, merge_percent=None, level_selector='median', use_maximums=True,
                 bars_for_peak=21):

        self._use_max = use_maximums
        self._bars_for_peak = bars_for_peak
        super().__init__(merge_distance, merge_percent, level_selector)

    def _validate_init_args(self):
        super()._validate_init_args()
        if self._bars_for_peak % 2 == 0:
            raise Exception('N bars to define peak should be odd number')

    def _find_potential_level_prices(self, X):
        d = pd.DataFrame(data=X, columns=('price',))
        bars_to_shift = int((self._bars_for_peak - 1) / 2)

        if self._use_max:
            d['F'] = d['price'].rolling(window=self._bars_for_peak).max().shift(-bars_to_shift)
        else:
            d['F'] = d['price'].rolling(window=self._bars_for_peak).min().shift(-bars_to_shift)

        prices = pd.unique(d[d['F'] == d['price']]['price'])

        return prices

    def _aggregate_prices_to_levels(self, prices, distance):
        return _cluster_prices_to_levels(prices, distance, self._level_selector)


df = pd.read_csv("candles.txt")
cl = RawPriceClusterLevels(None, merge_percent=0.25, use_maximums=True, bars_for_peak=91)
cl.fit(df)

levels = []
for level in cl.levels:
    levels.append(level['price'])

levels.sort()

with open("levels.txt", 'w') as fp:
    fp.write('\n'.join(str(level) for level in levels))
