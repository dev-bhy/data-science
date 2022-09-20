import os
import sys
import errno
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import optuna
import plotly
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import numpy as np 
# from src.methods.mrmr_feature_selection import mrmr_feature_selection
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from boruta_shap import run_boruta_shap
# from mimic_fs import run_mimic
# from cfs_v2 import run_cfs
from data_utils import preprocessing, read_partial_blobs
import pandas as pd
from datetime import datetime

def feature_selector(trial, method, x_train, y_train):
    if method == 'mrmr':
        pass
    elif method == 'relief':
        pass
    elif method == 'r_relief':
        pass
    elif method == 'aco':
        pass
    elif method == 'boruta_shap':
        model = 'rf'
        p_value = trial.suggest_float("p_value", 0.2, 0.6, step = 0.05)
        numTrials =  trial.suggest_int("numTrials", 200, 250, step = 10)
        perc = trial.suggest_int("perc", 20, 50, step = 5)
        seed = 0
        features_list,_,_ = run_boruta_shap(x_train, y_train, model, perc = perc, p_value=p_value, numTrials=numTrials, seed = seed)
    elif method == 'mimic':
        threshold = trial.suggest_float("threshold",-0.01, 0.05, step = 0.005)
        features_list,_,_ = run_mimic(x_train, y_train, threshold= threshold)
    elif method == 'cfs':
        num_features = trial.suggest_int("num_features", 100,200, step = 2)
        opt, _ = run_cfs(x_train, y_train, num_features = num_features)

    return features_list

def split_train_val(x_train,y_train, num):
    return x_train[:-num], y_train[:-num], x_train[-num:], y_train[-num:]

def vif_calculation(x, thresh):
    tmp_vif = add_constant(x)
    vif_list = pd.Series([variance_inflation_factor(tmp_vif.values, i) 
        for i in range(tmp_vif.shape[1])], 
        index=tmp_vif.columns)

    print('original vif length: ', len(vif_list))
    cut_vif_list = vif_list
    # cut_vif_list = vif_list[vif_list != np.inf]
    # cut_vif_list = vif_list[vif_list < thresh]
    print('cut vif len: ', len(cut_vif_list))
    if (np.NaN in vif_list or np.NaN in vif_list):
        print("NAN in VIF")
        vif_mean = 99999
    else:
        vif_mean = np.mean(cut_vif_list.iloc[1:]) #The first row includes 'constant'

    return vif_list.iloc[1:], vif_mean, np.max(vif_list), list(cut_vif_list.index[1:])

def time_stamp():
    parent_dir = os.path.dirname(os.path.abspath(__file__)).strip('src') #Get parent directory
    now = datetime.now() #Get current time and date
    dir = parent_dir + 'data\\' + now.strftime("%m%d%Y") #Folder name is today's date (month, day, year)
    dt_string = now.strftime("%H%M%S") #Filename contains current time (hr,min,sec)
    print("date and time =", dt_string)
    try:
        os.makedirs(dir)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    try:
        os.makedirs(dir + '\\' + dt_string)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    return dir, dt_string

def remove_vars(train, test):
    for column in train.columns:
            if 'pca' in column:
                train = train.drop([column], axis = 1)
                test = test.drop([column], axis = 1)
    
    return train, test


def objective(trial, direc, interim_result, blobs_list, fs_method, eval_method, vif_thresh): 
    mae_lst_overblobs = []
    adj_r2_lst_overblobs = []
    r2_lst_overblobs=[]
    vif_lst_overblobs =[]
    df_list = []

    for blob in blobs_list:
        print(trial.number)
        # print(len(blobs_list))
        print("Process at: {} %".format((blobs_list.index(blob)+1)*100/len(blobs_list)))
        print(blob)
        file_name = blob.split("/")[-1]
        #Preprocessing divides data into train and test set
        x_train , y_train, x_test, y_test = preprocessing(file_name, verbose=False)
        #Remove PCA-ed variables
        # x_train, x_test = remove_vars(x_train, x_test)

        #Splits the training data into a new training and validation data
        #Set the num to pick the last 'num' rows as validation sets
        # x_train_new, y_train_new, x_val, y_val=split_train_val(x_train,y_train, num=10) 

        #Random split
        x_train_new, x_val, y_train_new, y_val = train_test_split(
            x_train, y_train, test_size=0.23, random_state=42
            )
        
        ######FEATURE SELECTION
        # Optimize Feature Selector 
        features_list = feature_selector(trial, fs_method, x_train_new, y_train_new)
        print("Reduced to : {}".format(100*len(features_list)/x_train.shape[1]))
        if (
            features_list == [] or features_list == [''] 
            or 
            len(features_list) < 0.1*(x_train_new.shape[1]) 
            # or
            # len(features_list) > 0.3 *(x_train_new.shape[1])
            ): #Prune it if the list returns empty.
            # raise optuna.TrialPruned()
            continue
        else:
            #Select features returned from the feature selection method
            x_train_new = x_train_new[features_list]    
            x_val_new = x_val[features_list]

            #Cut out features that have abnormally high VIF values
            # vif_list2, vif_mean, vif_max, cut_vif_lst = vif_calculation(x_train_new, vif_thresh)
            #If more than 60% of list was cut out, then prune this condition.
            # if (len(cut_vif_lst) < 0.4 *len(features_list) or len(cut_vif_lst)< 200):
            #     raise optuna.TrialPruned()
            # else:
            #     x_train_new = x_train_new[cut_vif_lst]
            #     x_val_new = x_val_new[cut_vif_lst]

        ######EVALUATION
        # Choose Evaluation Method
        if eval_method == 'lr':
            clf = LinearRegression() # Parameters to be added
            clf.fit(x_train_new.copy(), y_train_new.copy())
            val_pred = clf.predict(x_val_new.copy()) 
            train_pred = clf.predict(x_train_new.copy()) 

        elif eval_method == 'rf':
            clf = RandomForestRegressor(n_jobs = -1) # Parameters to be added
            clf.fit(x_train_new.copy(), y_train_new.copy())
            val_pred = clf.predict(x_val_new.copy()) 
            train_pred = clf.predict(x_train_new.copy()) 

        # mae 
        mae = mean_absolute_error(y_val, val_pred)
        print('mae', mae)
        mae_lst_overblobs.append(mae)

        #R^2
        r2 = r2_score(y_val, val_pred)
        print("r2: ", r2)
        r2_lst_overblobs.append(r2)

        # adjusted R^2
        adjusted_r2 = 1 - ((1 - r2_score(y_val, val_pred))*(x_val_new.shape[0] - 1)/(x_val_new.shape[0] - x_val_new.shape[1] - 1)) 
        print('adjusted r2:', adjusted_r2)
        adj_r2_lst_overblobs.append(adjusted_r2)
        # print(r2_lst_overblobs)

        # vif
        # vif_list, vif_mean, _ = vif_calculation(x_train_new, vif_thresh) 
        # vif_lst_overblobs.append(vif_mean)
        # print('vif: ',vif_mean)

        #Pruning condition for VIF
        # if (np.inf in vif_list):
        #     raise optuna.TrialPruned() 
        sku_score_list = [file_name] + [mae] + [r2] +[x_train_new.shape[1]] + [trial.params] #+ [vif_mean] + [vif_max] +[len(vif_list2)] + [len(vif_list2[vif_list2<vif_thresh])] 
        # print(sku_score_list)
        df_list.append(sku_score_list)
        # print(df_list)
        df_sku = pd.DataFrame(df_list, columns=['filename', 'MAE', 'r2', '# features', 'Trial Params']) #, 'VIF Mean', 'VIF Max','Num Features', 'Num High Vifs'])
        df_sku.to_csv('{}/{}_{}_interim_results.csv'.format(direc + '\\' + interim_result, trial.number,fs_method), index = False)

    return np.mean(mae_lst_overblobs)
    # return np.mean(r2_lst_overblobs), np.mean(vif_lst_overblobs)

def run_evaluation():
    ###SET INITIAL CONDITIONS
    dir, dt_string= time_stamp()
    blobs_list_tmp = read_partial_blobs(5,5,read_all = False)
    fs = 'boruta_shap'
    eval = 'lr'
    vf_thresh = 10**7

    ###START EVALUATIONS
    study = optuna.create_study(direction="minimize")
    # study = optuna.create_study(directions = ["maximize", "minimize"]) 
    study.optimize(lambda trial: objective(
        trial, direc = dir, interim_result = dt_string, blobs_list= blobs_list_tmp,
        fs_method =fs, eval_method = eval, vif_thresh = vf_thresh), n_trials=40)

    figure = optuna.visualization.plot_optimization_history(study)
    figure2 = optuna.visualization.plot_parallel_coordinate(study)
    figure3 = optuna.visualization.plot_slice(study)

    ###SAVE RESULTS
    df = study.trials_dataframe()
    dir_tmp2 = dir + '\\' + dt_string
    df.to_csv('{}\\{}_{}_tuning_results.csv'.format(dir_tmp2, dt_string, fs), index = False)
    figure.write_image("{}\\{}_{}_optim_history.png".format(dir_tmp2, dt_string, fs))
    figure2.write_image("{}\\{}_{}_parallel.png".format(dir_tmp2, dt_string, fs))
    figure3.write_image("{}\\{}_{}_slice.png".format(dir_tmp2, dt_string, fs))

run_evaluation()