import pandas as pd
import numpy as np
from time import time
import matplotlib.pyplot as plt

import itertools,sys

from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets.species_distributions import construct_grids
from sklearn.neighbors import KernelDensity
from sklearn import preprocessing

def get_weather_data_for_year():
    weather_data = pd.read_csv("../data/weather_data.csv")
    weather_data.rename(columns={"Weather_Date":"timestamp"}, inplace=True)
    return weather_data

class City:
    def __init__(self, x_left_lower_corner, y_left_lower_corner, x_upper_right_corner, 
                 y_upper_right_corner, grid_size=100):
    
        self.x_left_lower_corner = x_left_lower_corner
        self.y_left_lower_corner = y_left_lower_corner
        self.x_upper_right_corner = x_upper_right_corner
        self.y_upper_right_corner = y_upper_right_corner
        self.grid_size = grid_size
        self.dividers = self.construct_grid()

    
    def construct_grid(self):
        # x coordinates of the grid cells
        xgrid = np.linspace(self.x_left_lower_corner, self.x_upper_right_corner, self.grid_size)
        # y coordinates of the grid cells
        ygrid = np.linspace(self.y_left_lower_corner, self.y_upper_right_corner, self.grid_size)
        return (xgrid, ygrid)
    
    def get_map_coordinates(self):
        map_coordinates = []
        for i in itertools.product(self.dividers[0].tolist(),self.dividers[1].tolist()):
            map_coordinates.append(i)
        return map_coordinates
    
    def get_diagonals(self):
        map_coordinates = self.get_map_coordinates()
        n=self.grid_size
        diagonals = []
        for i in range(0, n*n - n-1):
            if(i!=0 and i%n==0):
                continue
            #print(i)
            diagonals.append((map_coordinates[i],map_coordinates[i+n+1]))
        return diagonals
        
def get_features(x):
    
    a_all = crime_df[(crime_df.timestamp == x.timestamp)&  (crime_df.lat >= x.cell_range[0][0])&
                                                   (crime_df.lat < x.cell_range[1][0])&
                                                   (crime_df.long >= x.cell_range[0][1])&
                                                   (crime_df.long < x.cell_range[1][1])].shape[0]
    
    a = crime_df[(crime_df.timestamp == x.timestamp)&  (crime_df.lat >= x.cell_range[0][0])&
                                                   (crime_df.lat < x.cell_range[1][0])&
                                                   (crime_df.long >= x.cell_range[0][1])&
                                    (crime_df.long < x.cell_range[1][1])].groupby('ctype').count().reindex(ctypes).fillna(0)['timestamp'].values
    
    b = yelp_df[(yelp_df.lat >= x.cell_range[0][0])& (yelp_df.lat < x.cell_range[1][0])&
                                                   (yelp_df.long >= x.cell_range[0][1])&
                                                   (yelp_df.long < x.cell_range[1][1])].shape[0]
    
    c = police_df[(police_df.lat >= x.cell_range[0][0])&
                                               (police_df.lat < x.cell_range[1][0])&
                                               (police_df.long >= x.cell_range[0][1])&
                                               (police_df.long < x.cell_range[1][1])].shape[0]
    return [x.timestamp, x.cell_range,a_all, a,b,c]


def get_full_year_grid_wise_data(grid_line, grid_size):
    
    chicago = City(41.5487, -88.3713, 42.1176, -87.094, grid_line)
    cells = chicago.get_diagonals()
    #len(cells)

    all_index = np.array(list(itertools.product(pd.date_range(start=pd.datetime(2016, 1, 1), periods=366, freq='D').tolist(), cells)))
    timestamps = pd.to_datetime((all_index[:,:1]).flatten()).date
    grid_cells = all_index[:,1:].flatten()

    df = pd.DataFrame({'timestamp': timestamps, 'cell_range': grid_cells})
    
    featured_data = df.apply(lambda x: get_features(x), axis=1)
    
    temp = pd.DataFrame(featured_data,columns=['all_features'])

    temp[['timestamp','cell_range','crime_freq','each_crime','yelp_freq','police_freq']] = pd.DataFrame(temp['all_features'].values.tolist())

    temp[ctypes] = pd.DataFrame(temp['each_crime'].values.tolist())

    df = pd.merge(df, temp[['timestamp', 'cell_range', 'crime_freq', 'yelp_freq',
           'police_freq', 'theft', 'other', 'battery', 'assault', 'damage']], on=['cell_range','timestamp'], how='left')
    
    df.to_csv('../grids_full_year_%s.tsv'%grid_size, sep='\t', index=False)

grid_line = int(sys.argv[1])
grid_size = int(sys.argv[2])

yelp_df = pd.read_csv("../data/yelp_data.csv")
police_df = pd.read_csv("../data/police_data.csv")
crime_df = pd.read_csv("../data/crime_data.csv")
weather_df = get_weather_data_for_year()

theft_index = crime_df['Primary Type'].str.contains('THEFT')
battery_index = crime_df['Primary Type'].str.contains('BATTERY')
damage_index = crime_df['Primary Type'].str.contains('DAMAGE')
assault_index = crime_df['Primary Type'].str.contains('ASSAULT')
other_index = ~(theft_index | battery_index | damage_index | assault_index)

assert crime_df[theft_index].shape[0] + crime_df[battery_index].shape[0] + \
        crime_df[damage_index].shape[0] + crime_df[assault_index].shape[0] + crime_df[other_index].shape[0] == crime_df.shape[0]
        
crime_df['ctype'] = None

crime_df.loc[theft_index,'ctype'] = "theft"
crime_df.loc[battery_index,'ctype'] = "battery"
crime_df.loc[assault_index,'ctype'] = "assault"
crime_df.loc[damage_index,'ctype'] = "damage"
crime_df.loc[other_index,'ctype'] = "other"

assert crime_df.shape == crime_df.dropna().shape

ctypes = crime_df.ctype.unique()

print("Building")
get_full_year_grid_wise_data(grid_line, grid_size)
print("Saving")