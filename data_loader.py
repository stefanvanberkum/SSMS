"""
This module provides functions for loading and formatting the data.
"""

import os
from datetime import date

import numpy as np
import pandas as pd


def load_data(filepath: str):
    """
    Load the required data.

    :param filepath: the location of the files
    :return: formatted data
    """

    # Load supermarket transaction data.
    transaction_data = load_transaction(filepath)

    # Load turnover data.
    supermarket_data, hospitality_data = load_turnover(filepath)

    # Load tracker data.
    tracker_weekly = load_tracker(filepath)

    # Load temperature data.
    temperature_data = load_temperature(filepath)

    # Merge transaction, turnover, tracker and temperature data.
    data = transaction_data.merge(hospitality_data, left_index=True, right_index=True, how='left')
    data = data.merge(supermarket_data, left_index=True, right_index=True, how='left')
    data = data.merge(tracker_weekly, left_index=True, right_index=True, how='left')
    data = data.merge(temperature_data, left_index=True, right_index=True, how='left')

    # In this case, all regions are unique, for other partitions aggregation is required.
    data.index.name = 'YearWeek'
    data['Region'] = data['nuts3_code'] + '_' + data['StoreRegionNbr'].astype(str)

    # Aggregate rows by "Region".
    # Average sales per region is used, it's also possible to sum the sales per region.
    # StringencyIndex, Turnover_hospitality, and Turnover_supermarket are not region specific (only time dependent).
    # data = data.groupby([data.index, 'StoreRegionNbr', 'nuts3_code']).agg({
    #    'StoreRegionNbr': 'first', 'StoreRegionName': 'first', 'corop_name': 'first', 'nuts3_code': 'first',
    #    'QuantityCE': 'mean', 'SalesGoodsExclDiscountEUR': 'mean', 'SalesGoodsEUR': 'mean', 'NbrOfTransactions':
    #    'mean',
    #    'WVO': 'mean', 'SchoolHolidayMiddle': 'mean', 'SchoolHolidayNorth': 'mean', 'SchoolHolidaySouth': 'mean',
    #    'TempMin': 'mean', 'TempMax': 'mean', 'TempAvg': 'mean', 'RainFallSum': 'mean', 'SundurationSum': 'mean',
    #    '0-25_nbrpromos_index_201801': 'mean', '25-50_nbrpromos_index_201801': 'mean',
    #    '50-75_nbrpromos_index_201801': 'mean', 'Turnover_hospitality': 'mean', 'Turnover_supermarket': 'mean',
    #    'StringencyIndex': 'mean'})
    # data['Region'] = data.index.get_level_values('nuts3_code') + '_' + data.index.get_level_values(
    #    'StoreRegionNbr').astype(str)
    # data = data.reset_index(level=(1, 2), drop=True)

    # number_of_regions = data['Region'].nunique()
    # print(f'Number of unique regions: {number_of_regions}')
    # data.to_excel('data.xlsx')

    # # Plot variables, Groot-Rijnmond store 506 is only used as example
    # data_groot_rijnmond = data.loc[(data['corop_name'] == 'Groot-Rijnmond') & (data['StoreRegionNbr'] == 506)]
    # data_groot_rijnmond.reset_index().plot(x='YearWeek', y=['StringencyIndex'])
    # data_groot_rijnmond.reset_index().plot(x='YearWeek', y='SalesGoodsEUR')
    # data_groot_rijnmond.reset_index().plot.scatter(x='StringencyIndex', y='SalesGoodsEUR')
    # plt.show()

    # Drop incomplete time series.
    data = data[~data['Region'].isin(['NL332_330', 'NL327_330', 'NL212_509', 'NL423_340'])]
    return data


def load_transaction(filepath: str):
    """
    Loads the supermarket transaction data.

    :param filepath: the location of the supermarket transaction file
    :return: formatted data
    """

    # All the data is set to start from 2018-1-1 (201801) and uses YearWeek as index.
    transaction_data = pd.read_csv(os.path.join(filepath, 'transaction_data.csv'))

    # AH data for Belgium is dropped for now.
    transaction_data = transaction_data[
        (transaction_data['StoreRegionNbr'] != 610) & (transaction_data['StoreRegionNbr'] != 620) & (
                transaction_data['StoreRegionNbr'] != 630)]

    # Transform SchoolHolidayNorth, SchoolHolidayMiddle, SchoolHolidaySouth into 1 variable
    # This needs to happen before the index changes, otherwise the rows become indistinguishable
    nuts3_north = {'NL111', 'NL112', 'NL113', 'NL124', 'NL125', 'NL126', 'NL131', 'NL132', 'NL133', 'NL211', 'NL212',
                   'NL213', 'NL230', 'NL321', 'NL323', 'NL324', 'NL325', 'NL327', 'NL328', 'NL329'}
    nuts3_middle = {'NL225', 'NL221', 'NL224', 'NL310', 'NL332', 'NL333', 'NL337', 'NL33A', 'NL33B', 'NL33C'}
    nuts3_south = {'NL226', 'NL341', 'NL342', 'NL411', 'NL412', 'NL413', 'NL414', 'NL421', 'NL422', 'NL423'}
    transaction_data['SchoolHoliday'] = 0
    for index, row in transaction_data.iterrows():
        if row['nuts3_code'] in nuts3_north and row['SchoolHolidayNorth'] == 1:
            transaction_data.loc[index, 'SchoolHoliday'] = 1
        if row['nuts3_code'] in nuts3_middle and row['SchoolHolidayMiddle'] == 1:
            transaction_data.loc[index, 'SchoolHoliday'] = 1
        if row['nuts3_code'] in nuts3_south and row['SchoolHolidaySouth'] == 1:
            transaction_data.loc[index, 'SchoolHoliday'] = 1

    # Index by week of the year.
    transaction_data['WeekKey'] = transaction_data['WeekKey'].astype(str)
    transaction_data = transaction_data.set_index('WeekKey')

    # Take logs of SalesGoodsEUR, WVO, 0-25_nbrpromos_index_201801, 25-50_nbrpromos_index_201801, and
    # 50-75_nbrpromos_index_201801.
    to_log = ['SalesGoodsEUR', 'WVO', '0-25_nbrpromos_index_201801', '25-50_nbrpromos_index_201801',
              '50-75_nbrpromos_index_201801']
    for col in to_log:
        transaction_data[col] = np.log(transaction_data[col])
    return transaction_data


def load_turnover(filepath: str):
    """
    Loads the turnover data.

    :param filepath: the location of the turnover file
    :return: formatted data
    """

    # Load sector turnover data.
    turnover_data = pd.ExcelFile(os.path.join(filepath, 'hospitality_supermarket_turnover.xlsx'))

    # Process hospitality turnover data.
    hospitality_data = pd.read_excel(turnover_data, 'turnover_horeca_quarterly')

    # Remove last line (source disclaimer).
    hospitality_data = hospitality_data.iloc[:-1, :]

    # 2021Q4 is added to the last row, to upsample till first day of Q4 instead of first day of Q3
    hospitality_data.iloc[-1, hospitality_data.columns.get_loc('Perioden')] = '2021 4e kwartaal*'

    # Index by datetime in format %Y%W.
    hospitality_data['Perioden'] = pd.to_datetime(
        hospitality_data['Perioden'].str.replace(' ', '-Q', regex=False).str[:7])
    hospitality_data = hospitality_data[hospitality_data['Perioden'] >= '2018-01-01']
    hospitality_data = hospitality_data.set_index('Perioden')

    # Rename and format hospitality turnover variable.
    hospitality_data.rename(
        columns={'Seizoencorrectie_indexcijfers_omzet_waarde_basisjaar_2015': 'Turnover_hospitality'}, inplace=True)
    hospitality_data['Turnover_hospitality'] = pd.to_numeric(
        hospitality_data['Turnover_hospitality'].str.replace(',', '.'))

    # Use average over all different branches for seasonality-corrected revenue.
    hospitality_data = hospitality_data.groupby(hospitality_data.index).agg({'Turnover_hospitality': 'mean'})

    # Change quarterly data into weekly data by repeating the index.
    hospitality_data = hospitality_data.resample('W', convention='start').ffill()
    hospitality_data.index = hospitality_data.index.strftime('%G%V')

    # Process supermarket turnover data.
    supermarket_data = pd.read_excel(turnover_data, 'turnover_supermarkets_monthly')

    # Index by datetime in format %Y%W.
    months = {'januari': '01', 'februari': '02', 'maart': '03', 'april': '04', 'mei': '05', 'juni': '06', 'juli': '07',
              'augustus': '08', 'september': '09', 'oktober': '10', 'november': '11', 'december': '12'}
    for month in months:
        supermarket_data['maand'] = supermarket_data['maand'].str.replace(' ' + month, '-' + months[month] + '-01',
                                                                          regex=False)
    supermarket_data['maand'] = pd.to_datetime(supermarket_data['maand'].str[0:10], errors='coerce')
    supermarket_data = supermarket_data[supermarket_data['maand'] >= '2018-01-01']
    supermarket_data = supermarket_data.set_index('maand')

    # Rename supermarket turnover variable.
    supermarket_data.rename(
        columns={'Omzet_seizoengecorrigeerd_Indexcijfers_Waarde_basisjaar2015': 'Turnover_supermarket'}, inplace=True)

    # Change monthly data into weekly data by repeating the index.
    supermarket_data = supermarket_data.resample('W', convention='start').ffill()
    supermarket_data.index = supermarket_data.index.strftime('%G%V')
    return supermarket_data, hospitality_data


def load_tracker(filepath: str):
    """
    Loads the tracker data.

    :param filepath: the location of the tracker file
    :return: formatted data
    """

    # Load the tracker data.
    tracker = pd.read_csv(os.path.join(filepath, 'OxCGRT_latest.csv'), dtype={'RegionName': 'str', 'RegionCode': 'str'})

    # Filter by country and index by date.
    tracker = tracker.loc[tracker['CountryName'] == 'Netherlands']
    tracker['Date'] = pd.to_datetime(tracker['Date'], format='%Y%m%d')
    tracker = tracker.set_index('Date')

    # Pad with zeroes for missing dates (before COVID-19).
    tracker = tracker.reindex(pd.date_range(date(2018, 1, 1), tracker.index[-1]), fill_value=0)

    # Change daily data into weekly data by taking the mean.
    tracker_weekly = tracker.resample('W').agg(
        {'StringencyIndex': 'mean', 'GovernmentResponseIndex': 'mean', 'ContainmentHealthIndex': 'mean',
         'EconomicSupportIndex': 'mean'})
    tracker_weekly.index = pd.to_datetime(tracker_weekly.index, format='%Y-%m-%d').strftime('%G%V')
    return tracker_weekly


def load_temperature(filepath: str):
    """
    Loads the temperature data.

    :param filepath: the location of the temperature file
    :return: formatted data
    """

    # Load the temperature data.
    temperature_data = pd.read_excel(os.path.join(filepath, '24_hour_average_temperature.xlsx'))

    # Remove first 28 rows (information about data) and base header on 28th row without spaces/index label.
    header = temperature_data.iloc[27].str.strip()
    temperature_data = temperature_data[28:]
    temperature_data.columns = header
    temperature_data.columns.name = None

    # Index by date and remove data before 2018-1-1
    temperature_data['YYYYMMDD'] = pd.to_datetime(temperature_data['YYYYMMDD'], format='%Y%m%d')
    temperature_data = temperature_data[temperature_data['YYYYMMDD'] < '2018-01-01']

    # Transform temperature (TG) from string with spaces to numeric values (in 1.0 degrees Celsius).
    temperature_data = temperature_data.set_index('YYYYMMDD')
    temperature_data['TG'] = temperature_data['TG'].str.strip()
    temperature_data['TG'] = pd.to_numeric(temperature_data['TG'])
    temperature_data['TG'] = temperature_data['TG'].div(10)

    # Group by index by taking the mean
    temperature_data = temperature_data.groupby(temperature_data.index).mean()
    temperature_data.index = temperature_data.index.strftime('%m%d')

    # Group by day-month by taking the mean and remove leap day
    temperature_data = temperature_data.groupby(temperature_data.index).mean()
    temperature_data = temperature_data.drop('0229')

    # Add day-month averages of KNMI to the years 2018-2021
    old_temperature_data = temperature_data
    temperature_data = temperature_data.set_index('2018' + old_temperature_data.index)
    for i in range(3):
        temperature_data = temperature_data.append(old_temperature_data.set_index(str(2018 + i + 1)
                                                                                  + old_temperature_data.index))

    # Downsample daily data into weekly data by taking the mean
    temperature_data.index = pd.to_datetime(temperature_data.index, format='%Y%m%d')
    temperature_data = temperature_data.resample('W').mean()
    temperature_data.index = pd.to_datetime(temperature_data.index, format='%Y-%m-%d').strftime('%G%V')
    return temperature_data
