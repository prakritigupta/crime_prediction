import pandas as pd
import numpy as np
from time import time
import matplotlib.pyplot as plt

import itertools, datetime, sys

from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets.species_distributions import construct_grids
from sklearn.neighbors import KernelDensity
from sklearn import preprocessing

def get_normalizer(column):
    X = np.array(column).reshape(column.shape[0], 1)
    return preprocessing.Normalizer().fit(X)

def normalize(df, column_name):
    X = np.array(df[column_name]).reshape(df[column_name].shape[0], 1)
    return get_normalizer(df[column_name]).transform(X)

def get_scaler(column):
    X = np.array(column).reshape(column.shape[0], 1)
    return preprocessing.MinMaxScaler().fit(X)

def scale(df, column_name):
    X = np.array(df[column_name]).reshape(df[column_name].shape[0], 1)
    return get_scaler(df[column_name]).transform(X)

def normalize_columns(df, column_names):
    for column_name in column_names:
        df['%s_norm' % column_name] = normalize(df, column_name)
        
def scale_columns(df, column_names):
    for column_name in column_names:
        df['%s_scaled' % column_name] = scale(df, column_name)
        
def get_locations(df, metric='radians', drop_duplicates=True):
    if drop_duplicates:
        locations_df = df[['lat','long']].drop_duplicates()
    else:
        locations_df = df[['lat','long']]
    locations_array = np.vstack([locations_df['lat'],locations_df['long']]).T
    if metric == 'radians':
        locations_array *= np.pi / 180.  # Convert lat/long to radians
    return locations_array

def get_kde(df, drop_duplicates=True, bandwidth=0.00025, rtol=1E-4):
    locations = get_locations(df, drop_duplicates=drop_duplicates)
    #KDE initialization
    kde = KernelDensity(bandwidth=bandwidth, metric='haversine', kernel='gaussian',\
                        algorithm='ball_tree', rtol=rtol, atol=5)
    #fit with given location
#     print(locations.shape, drop_duplicates)
    kde.fit(locations)
    return kde

def get_weights(df, kernel, metric='radians', lat_label='lat', long_label='long'):
    locations_array = np.vstack([df[lat_label], df[long_label]]).T
    if metric == 'radians':
        locations_array *= np.pi / 180.  # Convert lat/long to radians
    if kernel == None:
        return None
    else:
        return np.exp(kernel.score_samples(locations_array))
        
def parse_location(loc):
    loc = loc.strip("()").split(',')
    lat = loc[0].strip()
    long = loc[1].strip()
    return float(lat), float(long)

def parse_cell_range(loc):
    loc = loc.strip("()").split('), (')
    ll = parse_location(loc[0].strip())
    ur = parse_location(loc[1].strip())
    return tuple(ll), tuple(ur)
    
def get_weather_data_for_year(year):
    weather_data = pd.read_csv("../data/PreProcessed_Weather_Data_%s.csv"%year)
    weather_data.rename(columns={"Weather_Date":"timestamp"}, inplace=True)
    return weather_data

    
def get_spatio_temporal_features(grid_size):
    print(grid_size)
    yelp_df = pd.read_csv("../data/Yelp.csv")
    police_df = pd.read_csv("../data/Police.csv")
    crime_df = pd.read_csv("../data/final_Crime.csv", parse_dates=['timestamp'])
    weather_df = get_weather_data_for_year(2016)

    print("Spatial")
    df = pd.read_csv('../grids_full_year_%s.tsv'%grid_size, sep='\t')
    df['lat'] = df.cell_range.apply(lambda x: (parse_cell_range(x)[0][0]+parse_cell_range(x)[1][0])/2)
    df['long'] = df.cell_range.apply(lambda x: (parse_cell_range(x)[0][1]+parse_cell_range(x)[1][1])/2)

    df['police_factor'] = get_weights(df, get_kde(police_df, bandwidth=0.0008))
    df['yelp_factor'] = get_weights(df, get_kde(yelp_df, bandwidth=0.0008, drop_duplicates=False))

    theft_kde = get_kde(crime_df[crime_df.ctype == "theft"][['lat','long']].reset_index(drop=True),\
                        bandwidth=0.0008, drop_duplicates=False)

    other_kde = get_kde(crime_df[crime_df.ctype == "other"][['lat','long']].reset_index(drop=True), \
                        bandwidth=0.0008, drop_duplicates=False)

    battery_kde = get_kde(crime_df[crime_df.ctype == "battery"][['lat','long']].reset_index(drop=True), \
                        bandwidth=0.0008, drop_duplicates=False)

    assault_kde = get_kde(crime_df[crime_df.ctype == "assault"][['lat','long']].reset_index(drop=True), \
                        bandwidth=0.0008, drop_duplicates=False)

    damage_kde = get_kde(crime_df[crime_df.ctype == "damage"][['lat','long']].reset_index(drop=True), \
                        bandwidth=0.0008, drop_duplicates=False)
                        
    df['theft_factor'] = get_weights(df, theft_kde)
    df['other_factor'] = get_weights(df, other_kde)
    df['battery_factor'] = get_weights(df, battery_kde)
    df['assault_factor'] = get_weights(df, assault_kde)
    df['damage_factor'] = get_weights(df, damage_kde)

    crime_kde = get_kde(crime_df, bandwidth=0.0008, drop_duplicates=False)
    df['crime_factor'] = get_weights(df, crime_kde)

    df.to_csv('../data/spatial/spatial_features_full_year_%s.tsv'%grid_size, sep='\t', index=False)
    
    print("Temporal")
    df = df.groupby(by=['cell_range','timestamp']).sum()

    #get previous day
    df['prev_day_crime_freq'] = df['crime_freq'].shift(1)
    df['prev_day_theft_freq'] = df['theft'].shift(1)
    df['prev_day_other_freq'] = df['other'].shift(1)
    df['prev_day_battery_freq'] = df['battery'].shift(1)
    df['prev_day_assault_freq'] = df['assault'].shift(1)
    df['prev_day_damage_freq'] = df['damage'].shift(1)

    #get previous week

    temp1 = df[['crime_freq','theft', 'other', 'battery', 'assault', 'damage']].shift(1)
    temp2 = df[['crime_freq','theft', 'other', 'battery', 'assault', 'damage']].shift(2)
    temp3 = df[['crime_freq','theft', 'other', 'battery', 'assault', 'damage']].shift(3)
    temp4 = df[['crime_freq','theft', 'other', 'battery', 'assault', 'damage']].shift(4)
    temp5 = df[['crime_freq','theft', 'other', 'battery', 'assault', 'damage']].shift(5)
    temp6 = df[['crime_freq','theft', 'other', 'battery', 'assault', 'damage']].shift(6)
    temp7 = df[['crime_freq','theft', 'other', 'battery', 'assault', 'damage']].shift(7)
    
    df[['prev_7_days_crime_freq','prev_7_days_theft_freq','prev_7_days_other_freq',\
    'prev_7_days_battery_freq','prev_7_days_assault_freq','prev_7_days_damage_freq']] = (temp1+temp2+temp3+temp4+temp5+temp6+temp7)/7
    
    df.reset_index(drop=False, inplace=True)
    
    #remove NaN and set it back to normal  index
    df = df[pd.to_datetime(df.timestamp) > datetime.date(2016,1,7)].reset_index(drop=True)

    df.to_csv('../data/spatio_temporal/spatial_temporal_features_full_year_%s.tsv'%grid_size, sep='\t', index=False)
    
    
grid_size = int(sys.argv[1])

get_spatio_temporal_features(grid_size)