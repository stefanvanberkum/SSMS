import pandas as pd
from datetime import date
import matplotlib.pyplot as plt

transaction_data = pd.read_csv('AlbertHeijn_TranactionData.csv')
# AH data for Belgium is dropped for now
transaction_data = transaction_data[(transaction_data['StoreRegionNbr'] != 610)
                                    & (transaction_data['StoreRegionNbr'] != 620)
                                    & (transaction_data['StoreRegionNbr'] != 630)]
transaction_data['WeekKey'] = pd.to_datetime(transaction_data['WeekKey'].astype(str) + '0', format='%Y%W%w')
# WeekKey is used as index of the dataframe, same index is used for tracker data
transaction_data = transaction_data.set_index('WeekKey')
transaction_data.index = pd.to_datetime(transaction_data.index, format='%Y-%m-%d').strftime('%Y%W')

tracker = pd.read_csv('OxCGRT_latest.csv')
tracker = tracker.loc[tracker['CountryName'] == 'Netherlands']
tracker['Date'] = pd.to_datetime(tracker['Date'], format='%Y%m%d')
tracker = tracker.set_index('Date')
# Add data with 0's from 1-1-2018 till first date of tracker
tracker = tracker.reindex(pd.date_range(date(2018, 1, 1), tracker.index[-1]), fill_value=0)
# Change daily data into weekly data
tracker_weekly = tracker.resample('W').agg({'StringencyIndex': 'mean',
                                            'GovernmentResponseIndex': 'mean',
                                            'ContainmentHealthIndex': 'mean',
                                            'EconomicSupportIndex': 'mean'})
tracker_weekly.index = pd.to_datetime(tracker_weekly.index, format='%Y-%m-%d').strftime('%Y%W')

# Merge AH data with tracker data
data = transaction_data.merge(tracker_weekly, left_index=True, right_index=True, how='left')

# Aggregate rows by "Region"
data.index.name = 'YearWeek'
data = data.groupby([data.index, 'StoreRegionNbr', 'nuts3_code']).agg({
    'StoreRegionNbr': 'first', 'StoreRegionName': 'first', 'corop_name': 'first', 'nuts3_code': 'first',
    'QuantityCE': 'mean', 'SalesGoodsExclDiscountEUR': 'mean', 'SalesGoodsEUR': 'mean', 'NbrOfTransactions': 'mean',
    'WVO': 'mean', 'SchoolHolidayMiddle': 'mean', 'SchoolHolidayNorth': 'mean', 'SchoolHolidaySouth': 'mean',
    'TempMin': 'mean', 'TempMax': 'mean', 'TempAvg': 'mean', 'RainFallSum': 'mean', 'SundurationSum': 'mean',
    '0-25_nbrpromos_index_201801': 'mean', '25-50_nbrpromos_index_201801': 'mean',
    '50-75_nbrpromos_index_201801': 'mean', 'StringencyIndex': 'mean'})
data['Region'] = data.index.get_level_values('nuts3_code') + '_' + data.index.get_level_values('StoreRegionNbr').map(str)
data = data.reset_index(level=(1, 2), drop=True)
# number_of_regions = data['Region'].nunique()
# print(f'Number of unique regions: {number_of_regions}')
# data.to_excel('data.xlsx')

# Plot variables, Groot-Rijnmond store 506 is only used as example
# data_groot_rijnmond = data.loc[(data['corop_name'] == 'Groot-Rijnmond') & (data['StoreRegionNbr'] == 506)]
# data_groot_rijnmond.reset_index().plot(x='YearWeek', y=['StringencyIndex'])
# data_groot_rijnmond.reset_index().plot(x='YearWeek', y='SalesGoodsEUR')
# data_groot_rijnmond.reset_index().plot.scatter(x='StringencyIndex', y='SalesGoodsEUR')
# plt.show()
