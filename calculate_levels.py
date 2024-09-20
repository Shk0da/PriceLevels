import sys
sys.path.insert(1, 'pricelevels')
import pandas as pd

from cluster import RawPriceClusterLevels
from levels_on_candlestick import plot_levels_on_candlestick

df = pd.read_csv(sys.argv[1])
cl = RawPriceClusterLevels(None, merge_percent=0.25, use_maximums=True, bars_for_peak=91)
cl.fit(df)

levels = []
for level in cl.levels:
    levels.append(level['price'])

levels.sort()

with open(sys.argv[2], 'w') as fp:
    fp.write('\n'.join(str(level) for level in levels))
