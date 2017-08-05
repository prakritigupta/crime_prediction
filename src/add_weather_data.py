import pandas as pd
import numpy as np
import sys

def get_weather_data_for_year(year):
    weather_data = pd.read_csv("../data/PreProcessed_Weather_Data_%s.csv"%year)
    weather_data.rename(columns={"Weather_Date":"timestamp"}, inplace=True)
    return weather_data


weather_df = get_weather_data_for_year(2016)

grid_size = int(sys.argv[1])
feature_df = pd.read_csv('../data/spatio_temporal/spatial_temporal_features_full_year_%s.tsv'%grid_size, sep='\t')

new_feature_df = pd.merge(feature_df, weather_df, on='timestamp', how='left')

new_feature_df.to_csv('../data/spatio_temporal/full_features_year_with_weather_%s_final.tsv'%grid_size, sep='\t', index=False)
