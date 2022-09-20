import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
import numpy as np
import pandas as pd
from src.data_utils import preprocessing,read_partial_blobs
from minepy import MINE
import minepy
import time
import random
from joblib import Parallel, delayed
import progressbar
from time import sleep
from datetime import datetime

def calculate_mic(feature1, label, switch):
    if switch == 0:
        mine = MINE(alpha = 0.6, c =15, est = "mic_approx")
    elif switch == 1:
        mine = MINE(alpha = 0.6, c = 15, est = "mic_e")
    mine.compute_score(feature1,label)
    return mine.mic()#, mine.tic(), mine.get_score()

def get_relevance(X,y):
    i = 0
    subset_idx = []
    subset_mic = []
    df_subset = pd.DataFrame()
    for feature in X.columns:
        a= calculate_mic(X[feature], y, 1)
        if True: #a>threshold
            subset_idx.append(i)
            subset_mic.append(a)
        # print(a)
        i+=1
    df_subset['features_idx'] = subset_idx
    df_subset['features'] = X.columns[subset_idx]
    df_subset['mic_e'] = subset_mic
    F_order = df_subset.sort_values(by=['mic_e'], ascending = False)

    return F_order

def R_fi_subset(X, fi, subset):
    res = 0
    for feature_name in subset:
        res += calculate_mic(X[fi],X[feature_name],1)
    if subset == []:
        return 0
    else:
        return res/len(subset)

def miMic(X, fi,subset, df_FC):
    #fi : feature name
    #subset: Optimal subset
    #df_FC: Ordered list of mic_e
    MIC_fiC = float(df_FC[df_FC.features == fi].mic_e)
    R = R_fi_subset(X, fi, subset)
    return MIC_fiC - R

def R_fi_subset_parallel(X, fi, subset):
    res = 0
    res = np.sum(Parallel(n_jobs=-1, prefer = 'threads')(
        delayed(calculate_mic)(X[fi], X[feature_name], 1
                      ) for feature_name in subset))
    # for feature_name in subset:
    #     res += calculate_mic(X[fi],X[feature_name],1)
    if subset == []:
        return 0
    else:
        return res/len(subset)

def miMic_parallel(X, fi,subset, df_FC):
    #fi : feature name
    #subset: Optimal subset
    #df_FC: Ordered list of mic_e
    MIC_fiC = float(df_FC[df_FC.features == fi].mic_e)
    R = R_fi_subset_parallel(X, fi, subset)
    return MIC_fiC - R

def run_mimic(X,y, threshold = 0):
    time1 = time.time()
    # F_order_df = F_order.copy()
    F_order_df = get_relevance(X,y)
    F_order_features = list(F_order_df.features) #Input
    num_original_features = len(F_order_features)
    F_opt = []

    # while F_.isempty()== False:
    bar = progressbar.ProgressBar(maxval=num_original_features, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    j = 0

    for feature in F_order_features:
        # j/(len(F_order_features))
        # print(feature)
        mimic = miMic_parallel(X, feature, F_opt, F_order_df)
        # print(mimic)
        if mimic < threshold:
            pass
        else: 
            # print('th')
            F_opt.append(feature)

        bar.update(j+1)
        # sleep(0.1)
        j+=1

    bar.finish()
    walltime = time.time() - time1
    # os.system('say "done"')

    return F_opt, len(F_opt), walltime

# all_blobs = read_partial_blobs(10,10,True)

# df_list = []
# now = datetime.now()

# print("now =", now)

# # YYmmdd_HMS
# dt_string = now.strftime("%Y%m%d_%H%M%S")
# print("date and time =", dt_string)

# for path in all_blobs:
# # if True:
# #     path = all_blobs[1]
#     print("Process at: {} %".format((all_blobs.index(path)+1)*100/len(all_blobs)))
#     file_name = path.split("/")[-1]

#     x_train , y_train, x_test, y_test = preprocessing(file_name, True)

#     opt, num_opt, wltime = run_mimic(x_train, y_train, threshold = 0.05)

#     conc_lst = [path] + [opt] + [wltime] + [len(x_train.columns), len(x_train)] + [num_opt]

#     df_list.append(conc_lst)
#     # os.system('say "done"')
#     os.system('printf \a')

#     df = pd.DataFrame(df_list, columns= ['filepath', 'accepted', 'walltime', 'num input features', 'num rows', 'num output features'])
#     df.to_csv('{}_micmic_walltime.csv'.format(dt_string), index = False)