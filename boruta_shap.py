import os
import sys

# from data_utils import read_all_blobs
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import pandas as pd
import math
import os
import numpy as np

import xgboost
import lightgbm 
from sklearn.ensemble import RandomForestRegressor
from BorutaShap import BorutaShap

from src.data_utils import read_partial_blobs,traintest_split, chooseVars, read_data, preprocessing, calculate_similarity

import time
from datetime import datetime
# start_time = time.time()

def run_boruta_shap(x_train,y_train, model, perc, p_value, numTrials, seed):
    print('Boruta Shap initiated')

    if model.lower() == 'xgboost':
        md = xgboost.XGBRegressor()
    
    elif model.lower() == 'lgboost':
        md = lightgbm.LGBMRegressor(n_jobs = -1)
    
    elif model.lower() == 'rf':
        md = RandomForestRegressor(n_jobs = -1) #Default model is RandomForest
    
    print("Model: {}, percentile: {}, p-value: {}, and numTrials: {}".format(model, perc,p_value, numTrials))
    try:

        #Generate BorutaShap classifier
        FS = BorutaShap(model= md, importance_measure='Shap',
                    classification=False, percentile=perc, pvalue=p_value)

        #Fit Boruta shap (Best optimized for randomforest)
        FS.fit(X= x_train, y= y_train, n_trials = numTrials, random_state=seed, sample=False,
                train_or_test = 'train', normalize=True, verbose=False, stratify=None)  
        # FS.accepted == []
        # print('Accepted list is empty. Please modify your hyperparameters for BorutaShap')
        return FS.accepted, FS.tentative, FS.rejected

    except:
        return [], [], []

def test_over_allSKU():
    df_list = []
    all_blobs = read_partial_blobs(_,_,True)

    # datetime object containing current date and time
    now = datetime.now()

    print("now =", now)

    # YYmmdd_HMS
    dt_string = now.strftime("%Y%m%d_%H%M%S")
    print("date and time =", dt_string)

    # print(all_blobs[78])
    for path in all_blobs:
    # if True:
    #     path = all_blobs[1]
        print("Process at: {} %".format((all_blobs.index(path)+1)*100/len(all_blobs)))
        file_name = path.split("/")[-1]

        x_train , y_train, x_test, y_test = preprocessing(file_name, True)
        """
        without CV
        """
        # x_val = x_train.iloc[-10:]
        # y_val = y_train.iloc[-10:]
        x_train = x_train.iloc[:-10]
        y_train = y_train.iloc[:-10]
        """
        """
        time1 = time.time()
        accepted, _, _ = run_boruta_shap(x_train,y_train, 'rf', 0.05, 100, seed = 0)

        walltime = time.time() - time1

        conc_lst = [path] + [accepted] + [walltime] + [len(x_train.columns), len(x_train)]

        df_list.append(conc_lst)
        os.system('say "done"')

        df = pd.DataFrame(df_list, columns= ['filepath', 'accepted', 'walltime', 'num input features', ''])
        df.to_csv('{}_boruta_walltime.csv'.format(dt_string), index = False)

def test_stability(num_iter_range, seed_list):
    df_list = []
    now = datetime.now()

    print("now =", now)

    # YYmmdd_HMS
    dt_string = now.strftime("%Y%m%d_%H%M%S")

    stability_score = []
    for current_seed in seed_list:
        hist_accepted = []
        for i in range(num_iter_range):
            path = read_partial_blobs(_,_,read_all = True)[1]
            file_name = path.split("/")[-1]

            x_train , y_train, x_test, y_test = preprocessing(file_name, True)

            accepted, _, _ = run_boruta_shap(x_train,y_train, 'rf', 0.1, 100, seed = current_seed)

            if accepted == []:
                raise('accepted is empty')
            else:
                print(accepted)
                hist_accepted.append(accepted)
        
        for j in range(num_iter_range):
            if j == 0:
                stability_score.append(1)
            else:
                stability_score.append(calculate_similarity(hist_accepted[0], hist_accepted[j], 'jaccard'))
        
        conc_list = [current_seed] + [stability_score] + [seed_list] +[num_iter_range]
        df_list.append(conc_list)

        df = pd.DataFrame(df_list, columns=['seed', 'score', 'seed list', 'iterations'])
        df.to_csv('{}_boruta_stability.csv'.format(dt_string), index = False)
            


# test_stability(3, [30,42, 50, 60, 120, 2000])
                




