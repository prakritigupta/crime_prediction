import pandas as pd
import numpy as np
from time import time
import matplotlib.pyplot as plt, ast

import itertools, datetime, time, multiprocessing,sys

from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets.species_distributions import construct_grids
from sklearn.neighbors import KernelDensity
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import svm
from sklearn import neural_network,  neighbors

from sklearn.model_selection import TimeSeriesSplit

import warnings
warnings.filterwarnings('ignore')

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

def SMAPE_mod(actual, predicted):
    length = len(actual)
    smape_mod = 0
    for i in range(length):
        #if any one of them is not equal to zero
        if (actual[i] != 0) or (predicted[i] != 0):
            smape_mod = smape_mod + np.abs(actual[i] - predicted[i])/(actual[i] + predicted[i])
    smape_mod = (smape_mod/length)*100
    return smape_mod
    
class Scale_data:
    def __init__(self, columns=None, output = None):
        self.columns = columns
        self.output = output
    
    def scale_data_with_columns(self, X,y):
        self.scaler_X = preprocessing.MinMaxScaler()
        self.scaler_X.fit(X)
        self.scaled_X = pd.DataFrame(self.scaler_X.transform(X), columns=self.columns)
        self.scaler_y = preprocessing.MinMaxScaler()
        self.scaler_y.fit(y.values.reshape(-1,1))
        self.scaled_y = pd.DataFrame(self.scaler_y.transform(y.values.reshape(-1,1)), columns=[self.output])
        return self.scaled_X, self.scaled_y
    
    def scale_X_with_columns(self, X,):
        self.scaler_X = preprocessing.MinMaxScaler()
        self.scaler_X.fit(X)
        self.scaled_X = pd.DataFrame(self.scaler_X.transform(X), columns=self.columns)
        return self.scaled_X
    
    
    def unscale_data(self, scaled_y):
        return self.scaler_y.inverse_transform(scaled_y.reshape(-1,1))

def train_and_validate(X_train, X_test, y_train, y_test, output, features=None, model = None, log=False):
    '''
        Pass features with the columns with which we want to test our model
    '''
    # let's scale the train data
    train_scaler = Scale_data(features, output)
    scaled_X_train, scaled_y_train = train_scaler.scale_data_with_columns(X_train, y_train)

    
    # scale the test data using train scaler
    scaled_X_test, scaled_y_test = train_scaler.scale_data_with_columns(X_test, y_test)

    start_time = time.time()
    
    model.fit(scaled_X_train, scaled_y_train.values.flatten())
    end_time = time.time()
    scaled_y_test_pred = model.predict(scaled_X_test)

    #unscale output data using same training scale
    y_test_pred = train_scaler.unscale_data(scaled_y_test_pred)
    
    #return predicted result and training time
    return y_test_pred, (end_time - start_time), model

def run_validation_CV(data, columns_used, output, model):
    detail_list = []

    for i in range(2,12):
        train_X = data[data.index.isin(range(1,i+1))][columns_used]
        train_y = data[data.index.isin(range(1,i+1))][output]
        validation_X = data[data.index == (i+1)][columns_used]
        validation_y = data[data.index == (i+1)][output]
                   
        y_pred, train_time, fitted_model = train_and_validate(train_X, validation_X, train_y, validation_y, output, features=columns_used, model = model)
        y_pred = y_pred.clip(min=0)
        rmse = np.sqrt(metrics.mean_squared_error(validation_y.values.flatten(), y_pred))
        detail_list.append([i,train_time, rmse])
        
    df = pd.DataFrame(detail_list, columns=['trainingSize','trainingTime',"RMSE"])
    #return mean error
    return df.mean()['RMSE']

def find_one_SVR_parameter_all_CV(data, columns_used, output, grid_size, feature_elim=False):
    SVR_model_eval = []
    if feature_elim == False:
        k = "all"
        for c,epsilon in itertools.product([10,5,1,0.75, 0.5,0.1], [0.01,0.05,0.1]):
            model = svm.SVR(C=c, epsilon=epsilon, cache_size=4000)
            #now run it for all CV and find average error
            error = run_validation_CV(data, columns_used, output, model)
            #add error
            #print(error)
            SVR_model_eval.append([c, epsilon, k, error])
    else:
        for c,epsilon,k in itertools.product([10,5,1,0.75, 0.5,0.1], [0.01,0.05,0.1],\
                                                        range(1,len(columns_used))):
            model = svm.SVR(C=c, epsilon=epsilon, cache_size=4000)
            #now run it for all CV and find average error
            error = run_validation_CV(data, columns_used[0:k], output, model)
            #add error
            SVR_model_eval.append([c, epsilon,k, error])

    SVR_model_eval = pd.DataFrame(SVR_model_eval, columns=['C','epsilon',"k",'RMSE']).groupby(by=['C', 'epsilon','k']).sum()
    
    SVR_model_eval.to_csv("../results/SVR/%s/%s_param_CV_error.csv"%(grid_size,output))
    print(SVR_model_eval.RMSE.argmin())
    best_C, best_epsilon,best_k = SVR_model_eval.RMSE.argmin()
    return best_C, best_epsilon,best_k


def get_MLP_parameters_all_CV(data, columns_used, output, grid_size,feature_elim=False):
    MLP_model_eval = []
    if feature_elim == False:
        k = "all"
        for number_hidden_layer in range(1,30):
            model = neural_network.MLPRegressor(hidden_layer_sizes=number_hidden_layer,early_stopping=True,tol=1e-2,\
                                               shuffle=False, max_iter=10000, solver='sgd', random_state=4,warm_start=True)
            #now run it for all CV and find average error
            error = run_validation_CV(data, columns_used, output, model)

            #Calculate R^2
            MLP_model_eval.append([number_hidden_layer, k, error])
            
    else:
        for number_hidden_layer,k in itertools.product(range(1,30), range(1,len(columns_used))):
            model = neural_network.MLPRegressor(hidden_layer_sizes=number_hidden_layer, early_stopping=True, tol=1e-2,
                                                shuffle=False, max_iter=10000, solver='sgd', random_state=4, warm_start=True)
            #now run it for all CV and find average error
            error = run_validation_CV(data, columns_used[0:k], output, model)

            #Calculate R^2
            MLP_model_eval.append([number_hidden_layer, k, error])
        
    MLP_model_eval = pd.DataFrame(MLP_model_eval, columns=['Hidden Layers','k','RMSE']).groupby(by=['Hidden Layers','k']).sum()
    MLP_model_eval.to_csv("../results/MLP/%s/%s_param_CV_error.csv"%(grid_size,output))   
    print(MLP_model_eval.RMSE.argmin())
    best_hidden_layer, best_k = MLP_model_eval.RMSE.argmin()
    return best_hidden_layer, best_k
    
    
def find_one_LRSGD_parameter_all_CV(data, columns_used, output, grid_size,feature_elim=False):
    SGD_model_eval = []
    print("Feature ELim: ", feature_elim)
    if feature_elim == False:
        k= 'all'
        for penalty,alpha,lr,eta0 in itertools.product(['l1','l2','elasticnet'],[0.1,0.001,0.01,0.0001],\
                                                    ['constant','optimal'],[1,0.1,0.001,0.01,0.0001]):
            model = linear_model.SGDRegressor(penalty=penalty, alpha=alpha, learning_rate=lr, \
                                          eta0=eta0, random_state=4, shuffle=False)
            
            #now run it for all CV and find average error
            error = run_validation_CV(data, columns_used, output, model)
            SGD_model_eval.append([penalty,alpha,lr,eta0, k, error])
        
    SGD_model_eval = pd.DataFrame(SGD_model_eval,\
                              columns=['penalty','alpha','lr','eta0','k','RMSE']).groupby(by=['penalty','alpha','lr','eta0','k']).sum()
    print(SGD_model_eval.RMSE.argmin())
    SGD_model_eval.to_csv("../results/SGD/%s/%s_param_CV_error.csv"%(grid_size,output))  
    return SGD_model_eval.RMSE.argmin()
    
def find_one_DecisionTree_parameter_all_CV(data, columns_used, output, grid_size,feature_elim=False):
    DT_model_eval = []
    if feature_elim == False:
        k = "all"
        for msl,mss,md,mf in itertools.product([0.2,0.5], [0.1,2],[2,None],[0.1,0.2,0.4,'sqrt','log2',None]):
            model = DecisionTreeRegressor(criterion="mse", random_state=4, min_samples_leaf=msl, min_samples_split=mss,\
            max_depth=md, max_features=mf)
            #now run it for all CV and find average error
            error = run_validation_CV(data, columns_used, output, model)

            #Calculate R^2
            DT_model_eval.append([msl,mss,md,mf, k,error])
        
    DT_model_eval = pd.DataFrame(DT_model_eval, columns=['msl','mss','md','mf','k','RMSE']).groupby(by=['msl','mss','md','mf','k']).sum()
    DT_model_eval.to_csv("../results/DecisionTree/%s/%s_param_CV_error.csv"%(grid_size,output))   
    print(DT_model_eval.RMSE.argmin())
    best_msl, best_mss, best_md, best_mf, best_k = DT_model_eval.RMSE.argmin()
    return best_msl,best_mss,best_md,best_mf, best_k

def get_kNN_parameters_all_CV(data, columns_used, output, grid_size,feature_elim=False):
    kNN_model_eval = []
    if feature_elim == False:
        k = "all"
        for n,w,p, leaf in itertools.product(range(5,15), ['uniform','distance'],[1,2], range(10,21)):
            model = neighbors.KNeighborsRegressor(n_neighbors=n, weights=w, p=p, n_jobs=-1, leaf_size = leaf)
            error = run_validation_CV(data, columns_used, output, model)
            kNN_model_eval.append([n,w,p,leaf, k, error])
    else:
        for n,w,p, leaf in itertools.product(range(5,15), ['uniform','balanced'],[1,2], range(10,21)):
            model = neighbors.KNeighborsRegressor(n_neighbors=n, weights=w, p=p, n_jobs=-1, leaf_size = leaf)
            error = run_validation_CV(data, columns_used[0:k], output, model)
            kNN_model_eval.append([n,w,p, leaf,k, error])

    kNN_model_eval = pd.DataFrame(kNN_model_eval, columns=['n','w','p','leaf','k','MSE']).groupby(by=['n','w','p','leaf','k']).sum()
      
    kNN_model_eval.to_csv("../results/kNN/%s/%s_param_CV_error.csv"%(grid_size,output))
    best_n, best_w, best_p, best_leaf,best_k = kNN_model_eval.MSE.argmin()
    return best_n, best_w, best_p, best_leaf,best_k

def train_and_test(X_train, X_test, y_train, y_test, grid_size, output, features, best_C, best_epsilon, \
                    model_list, best_hidden_layer, best_n, best_w, best_p, best_leaf, best_msl,best_mss,best_md,best_mf,\
                    best_penalty, best_alpha, best_lr,best_eta0):

    training_size = len(X_train.index.unique())
    
    X_train = X_train[features]
    X_test = X_test[features]
    
    scalerX = Scale_data(features)
    X_train = scalerX.scale_X_with_columns(X_train)
    X_test = scalerX.scale_X_with_columns(X_test)
    
    #scale_columns(X_train, X_test, features)
    yScaler = get_scaler(y_train)
    y_train = yScaler.transform(y_train)
    y_test = yScaler.transform(y_test)

    test = X_test.copy() 
    test['label'] = y_test
    
    if model_list == "LR":
        fit_model = linear_model.LinearRegression()

    if model_list == "SGD":
        fit_model = linear_model.SGDRegressor(penalty=best_penalty, alpha= best_alpha, learning_rate= best_lr, \
                                          eta0=best_eta0, random_state=4, shuffle=False)
                                          
    elif model_list == "DecisionTree":
        #print("here")
        fit_model = DecisionTreeRegressor(criterion="mse", random_state=4, min_samples_leaf=best_msl, min_samples_split=best_mss,\
            max_depth=best_md, max_features=best_mf)

    elif model_list == "SVR":
        fit_model = svm.SVR(C=best_C,epsilon=best_epsilon, cache_size=2000)
        
    elif model_list == "MLP":
        fit_model = neural_network.MLPRegressor(hidden_layer_sizes=best_hidden_layer, early_stopping=True,
                                                shuffle=False, max_iter=10000, solver='sgd', random_state=4)
                                                
    elif model_list == 'kNN':
        fit_model = neighbors.KNeighborsRegressor(n_neighbors=best_n, weights=best_w, p= best_p, n_jobs=-1, leaf_size = best_leaf)

    #record training time
    start_time = time.time()
    fit_model.fit(X_train, y_train)
    end_time = time.time()
    y_test_pred = fit_model.predict(X_test)

    y_test = yScaler.inverse_transform(y_test)

    y_test_pred = yScaler.inverse_transform(y_test_pred)
    #change <0  to 0
    y_test_pred = y_test_pred.clip(min=0)
    test['pred_label'] = y_test_pred
    test.to_csv('../results/%s/%s/%s_CV_%s.csv' % (model_list, grid_size,output,training_size), index=False)

    mae = metrics.mean_absolute_error(y_test, y_test_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))
    smape = SMAPE_mod(y_test, y_test_pred)
    #print(model_list+" RMSE: ",mae)
    #print(model_list+" MAE: ", rmse) 
    #print(model_list+ " SMAPE", smape)

    return mae, rmse, smape, (end_time-start_time)
    
def test_models(vector):
    features, output = vector
    model_list = models
    grid_size = grids
    print(vector, model_list, grid_size)
    verbose = False
    best_C = 0
    best_epsilon=0
    best_k=0
    best_hidden_layer = 0
    best_n = 0
    best_w = 0
    best_leaf=0
    best_p = 0
    best_msl = best_mss = best_md = best_mf = 0
    best_penalty = best_alpha = best_lr = best_eta0 = 0
    
    splits = 11
        
    if verbose:
        print()
        print('#'*100)
        print('#'*10 + str(output) + '#'*10)
        print('#'*100)
        print()

    X = df
    y = df[output]
    
    tscv = TimeSeriesSplit(n_splits=splits)

    if model_list == "SGD":
        best_penalty, best_alpha, best_lr,best_eta0, best_k = find_one_LRSGD_parameter_all_CV(X, features, output, grid_size)

    if model_list == "DecisionTree":
        best_msl,best_mss,best_md,best_mf, best_k = find_one_DecisionTree_parameter_all_CV(X, features, output, grid_size)   
    
    if model_list == "SVR":
        #find paramter for SVR before actually testing
        best_C, best_epsilon, best_k = find_one_SVR_parameter_all_CV(X, features, output, grid_size)
        
    if model_list == "MLP":
        best_hidden_layer, best_k = get_MLP_parameters_all_CV(X, features, output, grid_size)
        
    if model_list == "kNN":
        best_n, best_w, best_p, best_leaf, best_k = get_kNN_parameters_all_CV(X, features, output, grid_size)
            
    error_list = []
    for train_index, test_index in tscv.split(X.index.unique()):
        X_train, X_test = X[X.index.isin(train_index+1)], X[X.index.isin(test_index+1)]
        y_train, y_test = y[y.index.isin(train_index+1)], y[y.index.isin(test_index+1)]
    
        train_start_date, train_end_date = X_train.timestamp.values[0], X_train.timestamp.values[-1]
        test_start_date, test_end_date = X_test.timestamp.values[0], X_test.timestamp.values[-1]
        training_size = len(X_train.index.unique())
        
        if verbose:
            print()
            print('>'*100)
            print("Training: ", str(train_start_date), "to" , str(train_end_date))
            print("Test: ", str(test_start_date), "to" , str(test_end_date))
            print('>'*100)
            print()
        
        mae, rmse, smape,trainTime = train_and_test(X_train, X_test, y_train, y_test, grid_size, output, features, best_C, best_epsilon, model_list, best_hidden_layer,best_n, best_w, best_p, best_leaf, best_msl,best_mss,best_md,best_mf,\
        best_penalty, best_alpha, best_lr,best_eta0)
        
        error_list.append([train_index+1, mae, rmse, smape, trainTime])
        
    error_list = pd.DataFrame(error_list, columns=['Train Size','MAE','RMSE','SMAPE','trainTime'])
    error_list.to_csv("../results/%s/%s/%s_error.csv"%(model_list, grid_size, output))
    return error_list
    
feature_crime =  ['police_factor', 'yelp_factor', #'crime_factor',
       'prev_day_crime_freq',  'prev_7_days_crime_freq','PRCP', 'SNOW', 'TAVG', 'TMAX', 'TMIN']

feature_theft = ['PRCP', 'SNOW', 'TAVG', 'TMAX', 'TMIN',
       'police_factor', 'yelp_factor', #'theft_factor', 
       'prev_day_theft_freq', 'prev_7_days_theft_freq']

feature_other = ['PRCP', 'SNOW', 'TAVG', 'TMAX', 'TMIN',
       'police_factor', 'yelp_factor', #'other_factor',
       'prev_day_other_freq', 'prev_7_days_other_freq']

feature_battery = ['PRCP', 'SNOW', 'TAVG', 'TMAX', 'TMIN',
        'police_factor', 'yelp_factor', #'battery_factor', 
        'prev_day_battery_freq', 'prev_7_days_battery_freq']

feature_assault = ['PRCP', 'SNOW', 'TAVG', 'TMAX', 'TMIN',
       'police_factor', 'yelp_factor', #'assault_factor', 
       'prev_day_assault_freq', 'prev_7_days_assault_freq']

feature_damage = ['PRCP', 'SNOW', 'TAVG', 'TMAX', 'TMIN',
        'police_factor', 'yelp_factor', #'damage_factor', 
        'prev_day_damage_freq', 'prev_7_days_damage_freq']

if __name__ == "__main__":

    models = sys.argv[1]
    grids  = sys.argv[2]
    
    feature_type = sys.argv[3]

    df = pd.read_csv("../data/spatio_temporal/full_features_year_with_weather_%s_final.tsv"%grids, sep="\t")
    df.index = pd.to_datetime(df.timestamp).dt.month

    if feature_type == "all":
        argument_list = list(zip([feature_crime, feature_theft,feature_other,feature_battery,feature_assault,feature_damage],
                            ['crime_freq','theft','other','battery','assault','damage']))

    elif feature_type == "crime":
        argument_list = list(zip([feature_crime],['crime_freq']))
    #argument_list = list(zip([feature_crime, feature_damage],['crime_freq','damage']))


    p = multiprocessing.Pool(processes = len(argument_list), maxtasksperchild=1)
    print(len(argument_list))
    p.map(test_models, argument_list)