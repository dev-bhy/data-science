import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import errno

from scipy.stats import pearsonr
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import progressbar
import time
import pickle
from datetime import datetime
import cupy as cp

from src.data_utils import preprocessing, read_partial_blobs

def get_relevance(X,y):
    corr_list = []
    for num in range(len(X.columns)):
        corr1, _ = pearsonr(X[X.columns[num]], y)
        corr_list.append(corr1)

    df_pearson = pd.DataFrame()
    df_pearson['features'] = X.columns
    df_pearson['score'] = corr_list

    df_pearson_ordered = df_pearson.sort_values(by=['score'], ascending = False)

    return df_pearson_ordered

def merit(M, a, i , denom):
    # print("i: ",i)
    mat1 = np.abs(M[:a,:a])
    # print("mat1",mat1)
    # print("sum:mat1",np.sum(mat1))
    # print("sum:diag",np.sum(np.diag(mat1)))
    # print("M",M[i,:a])
    rff = (np.sum(mat1) - np.sum(np.diag(mat1)) + np.sum(np.abs(M[i,:a])))/(2*denom)
    # print("rff", rff)
    # print("left", np.abs(M[-1,:a]))
    # print("right", np.abs(M[-1,a]))
    rcf = (np.sum(np.abs(M[-1,:a])) + np.abs(M[-1,i]))/(a+1)
    # print(rcf)
    res = (a * rcf) / np.sqrt(a + a * (a-1) * rff)
    return res

def find_best_parallel(x_train, y_train, F_opt, F_order_features):
    merit_list = []

    a = len(F_opt)
    b = len(x_train)
    denom = a*(a+1)/2
    F_all = F_opt + F_order_features

    df_all = pd.concat([x_train[F_all], y_train], axis = 1)
    df_all_cols = df_all.columns

    # print(df_all_cols)
    M = np.corrcoef(df_all, rowvar=False)
    # print(M)
    merit_list.append(Parallel(n_jobs=-1, prefer = 'threads')(
            delayed(merit)(M, a, i, denom
                        ) for i in range(a, len(df_all_cols)-1)))
    best_index = np.argmax(merit_list)
    # print(best_index)
    best_col_name = F_order_features.pop(best_index-a)
    # print(best_col_name)
    F_opt.append(best_col_name)
    return F_opt, F_order_features

def run_cfs_v3(X,y, num_features):
    t1 = time.time()
    F_opt = []
    f_order = get_relevance(X, y) #Get relevancy matrix
    F_order_features = list(f_order.features) #Get feature names in descending order of score
    F_opt.append(F_order_features.pop(0)) 

    bar = progressbar.ProgressBar(maxval=num_features, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    j = 0

    while (len(F_opt) < num_features):
        F_opt, F_order_features = find_best_parallel(X, y, F_opt, F_order_features)
        # print(F_opt)
        bar.update(j+1)
        j+=1
    
    bar.finish()
    walltime = time.time() - t1

    return F_opt, walltime

all_blobs = read_partial_blobs(30,30,False)
parent_dir = os.path.dirname(os.path.abspath(__file__)).strip('src') #Get parent directory
now = datetime.now() #Get current time and date
dir = parent_dir + 'data\\' + now.strftime("%m%d%Y") + '\\cfs_results\\'  #Folder name is today's date (month, day, year)
try:
        os.makedirs(dir)
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise
    pass
for path in all_blobs[10:15]:
    print("Process at: {} %".format((all_blobs.index(path)+1)*100/len(all_blobs)))
    file_name = path.split("/")[-1]
    print(file_name)
    x_train , y_train, x_test, y_test = preprocessing(file_name, False)
    x_train = x_train.loc[:, (x_train != x_train.iloc[0]).any()]


    opt, wtime = run_cfs_v3(x_train, y_train, num_features = int(np.ceil(0.3*x_train.shape[1])) )
    # print(opt)
    with open(dir + path + '_cfs.pkl', 'wb') as f:
        pickle.dump(opt, f)

    with open(dir + path + '_cfs.pkl', 'rb') as f:
        mynewlist = pickle.load(f)

    print(mynewlist)
    print(wtime)

