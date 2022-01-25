import pandas as pd
from datetime import date
import matplotlib.pyplot as plt

# All the data is set to start from 2018-1-1 (201801) and uses YearWeek as index
transaction_data = pd.read_csv('AlbertHeijn_TranactionData.csv')
# AH data for Belgium is dropped for now
transaction_data = transaction_data[(transaction_data['StoreRegionNbr'] != 610)
                                    & (transaction_data['StoreRegionNbr'] != 620)
                                    & (transaction_data['StoreRegionNbr'] != 630)]
transaction_data['WeekKey'] = pd.to_datetime(transaction_data['WeekKey'].astype(str) + '0', format='%Y%W%w')
transaction_data = transaction_data.set_index('WeekKey')
transaction_data.index = pd.to_datetime(transaction_data.index, format='%Y-%m-%d').strftime('%Y%W')

turnover_data = pd.ExcelFile('Horeca_Supermarkt_Omzet.xlsx')
horeca_data = pd.read_excel(turnover_data, 'turnover_horeca_quarterly')
# 2021Q4 is added to the last row, to upsample till first day of Q4 instead of first day of Q3
horeca_data.iloc[-1, horeca_data.columns.get_loc('Perioden')] = '2021 4e kwartaal*'
horeca_data['Perioden'] = pd.to_datetime(horeca_data['Perioden'].str.replace(' ', '-Q').str[:7])
horeca_data = horeca_data[horeca_data['Perioden'] >= '2018-01-01']
horeca_data.rename(columns={'Seizoencorrectie_indexcijfers_omzet_waarde_basisjaar_2015': 'Turnover_horeca'}, inplace=True)
horeca_data['Turnover_horeca'] = pd.to_numeric(horeca_data['Turnover_horeca'].str.replace(',', '.'))
horeca_data = horeca_data.set_index('Perioden')
# Use average over all different Bedrijfstakken with seizoencorrectie
horeca_data = horeca_data.groupby(horeca_data.index).agg({'Turnover_horeca': 'mean'})
# Change quarterly data into weekly data
horeca_data = horeca_data.resample('W', convention='start').ffill()
horeca_data.index = pd.to_datetime(horeca_data.index, format='%Y-%m-%d').strftime('%Y%W')

# Turnover for supermarkets is not used for now
supermarket_data = pd.read_excel(turnover_data, 'turnover_supermarkets_monthly')
supermarket_data = supermarket_data.iloc[:-1, :]

tracker = pd.read_csv('OxCGRT_latest.csv')
tracker = tracker.loc[tracker['CountryName'] == 'Netherlands']
tracker['Date'] = pd.to_datetime(tracker['Date'], format='%Y%m%d')
tracker = tracker.set_index('Date')
tracker = tracker.reindex(pd.date_range(date(2018, 1, 1), tracker.index[-1]), fill_value=0)
# Change daily data into weekly data
tracker_weekly = tracker.resample('W').agg({'StringencyIndex': 'mean',
                                            'GovernmentResponseIndex': 'mean',
                                            'ContainmentHealthIndex': 'mean',
                                            'EconomicSupportIndex': 'mean'})
tracker_weekly.index = pd.to_datetime(tracker_weekly.index, format='%Y-%m-%d').strftime('%Y%W')

# Merge AH data with tracker data
data = transaction_data.merge(tracker_weekly, left_index=True, right_index=True, how='left')
data = data.merge(horeca_data, left_index=True, right_index=True, how='left')

# Aggregate rows by "Region"
# Average sales per region is used, it's also possible to sum the sales per region
# StringencyIndex and Turnover_horeca are not region specific (only time dependent)
data.index.name = 'YearWeek'
data = data.groupby([data.index, 'StoreRegionNbr', 'nuts3_code']).agg({
    'StoreRegionNbr': 'first', 'StoreRegionName': 'first', 'corop_name': 'first', 'nuts3_code': 'first',
    'QuantityCE': 'mean', 'SalesGoodsExclDiscountEUR': 'mean', 'SalesGoodsEUR': 'mean', 'NbrOfTransactions': 'mean',
    'WVO': 'mean', 'SchoolHolidayMiddle': 'mean', 'SchoolHolidayNorth': 'mean', 'SchoolHolidaySouth': 'mean',
    'TempMin': 'mean', 'TempMax': 'mean', 'TempAvg': 'mean', 'RainFallSum': 'mean', 'SundurationSum': 'mean',
    '0-25_nbrpromos_index_201801': 'mean', '25-50_nbrpromos_index_201801': 'mean',
    '50-75_nbrpromos_index_201801': 'mean', 'StringencyIndex': 'mean', 'Turnover_horeca': 'mean'})
data['Region'] = data.index.get_level_values('nuts3_code') + '_' + data.index.get_level_values('StoreRegionNbr').map(str)
data = data.reset_index(level=(1, 2), drop=True)
# number_of_regions = data['Region'].nunique()
# print(f'Number of unique regions: {number_of_regions}')
# data.to_excel('data.xlsx')

# # Plot variables, Groot-Rijnmond store 506 is only used as example
# data_groot_rijnmond = data.loc[(data['corop_name'] == 'Groot-Rijnmond') & (data['StoreRegionNbr'] == 506)]
# data_groot_rijnmond.reset_index().plot(x='YearWeek', y=['StringencyIndex'])
# data_groot_rijnmond.reset_index().plot(x='YearWeek', y='SalesGoodsEUR')
# data_groot_rijnmond.reset_index().plot.scatter(x='StringencyIndex', y='SalesGoodsEUR')
# plt.show()
