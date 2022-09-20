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

def getMerit_parallel(X, subset, label):
    # with cp.cuda.Device(0):
    k = len(subset)
    X_local = X[subset]
    X_local = np.asarray(X_local)
    label_np = np.asarray(label)
    
    # average feature-class correlation
    corr_mat_rcf = np.corrcoef(X_local, label_np, rowvar=False)
    corr_rcf = np.abs(corr_mat_rcf[-1][:-1])
    rcf = np.mean(corr_rcf)

    # average feature-feature correlation
    corr_mat = corr_mat_rcf[:-1,:-1]
    corr_data = np.triu(corr_mat, 1)
    abs_data = np.abs(corr_data)
    rff = np.mean(abs_data)

    res = (k * rcf) / np.sqrt(k + k * (k-1) * rff)
    return float(res)

def find_best_feature_parallel(x_train, y_train, F_opt, F_order_features):
    # t1 = time.time()
    lst_merit =[]
    lst_merit.append(Parallel(n_jobs=-1, prefer = 'threads')(
        delayed(getMerit_parallel)(x_train, F_opt + [feature], y_train
                      ) for feature in F_order_features))
    # print(lst_merit_gpu)
    idx = np.argmax(lst_merit)
    # print(type(float(idx)))
    F_opt.append(F_order_features.pop(int(idx)))
    # t2 = time.time()
    # print("time for np: ", t2-t1)

    return F_opt, F_order_features

def find_best_feature_gpu(x_train, y_train, F_opt, F_order_features):
    lst_merit =[]
    k = len(F_order_features)
    lst_merit_gpu = cp.zeros(k)
    t1 = time.time()
    i = 0
    for feature in F_order_features:
        this_set = F_opt + [feature]
        X_local = x_train[this_set]
        X_local_gpu = cp.asarray(X_local)
        label_gpu = cp.asarray(y_train)
        
        # average feature-class correlation
        corr_mat_rcf_gpu = cp.corrcoef(X_local_gpu, label_gpu, rowvar = False)
        corr_rcf_gpu = cp.abs(corr_mat_rcf_gpu[-1][:-1])
        rcf_gpu = cp.mean(corr_rcf_gpu)

        # average feature-feature correlation
        corr_mat_gpu = corr_mat_rcf_gpu[:-1,:-1]
        corr_data_gpu = cp.triu(corr_mat_gpu, 1)
        abs_data_gpu = cp.abs(corr_data_gpu)
        rff_gpu = cp.mean(abs_data_gpu)

        res =(k * rcf_gpu) / cp.sqrt(k + k * (k-1) * rff_gpu)
        cp.put(lst_merit_gpu, i, res)
        i+=1
    idx = cp.argmax(lst_merit_gpu)
    F_opt.append(F_order_features.pop(int(idx)))
    t2 = time.time()
    print("time for np: ", t2-t1)


    return F_opt, F_order_features

def find_best(x_train, y_train, F_opt, F_order_features):
    # print(F_opt)
    # print(F_order_features)
    np.set_printoptions(suppress=True)
    a = len(F_opt)

    denom = a*(a+1)/2
    F_all = F_opt + F_order_features

    df_all = pd.concat([x_train[F_all], y_train], axis = 1)
    df_all_cols = df_all.columns

    # print(df_all_cols)
    M = np.corrcoef(df_all, rowvar=False)
    # print(M)
    best_merit = 0
    for i in range(a, len(df_all_cols)-1):
        # print(i)
        mat1 = np.abs(M[:a,:a])
        # print("mat1",mat1)
        # print("sum:mat1",np.sum(mat1))
        # print("sum:diag",np.sum(np.diag(mat1)))
        # print("M",M[i,:a])
        rff = (np.sum(mat1) - np.sum(np.diag(mat1)) + np.sum(np.abs(M[i,:a])))/(2*denom)
        # print("rff", rff)
        # print("left", np.abs(M[-1,:a]))
        # print("right", np.abs(M[-1,a]))
        rcf = (np.sum(np.abs(M[-1,:a])) + np.abs(M[-1,a]))/(a+1)
        # print(rcf)
        res = (a * rcf) / np.sqrt(a + a * (a-1) * rff)
        # print("res", res)
        if res > best_merit:
            best_merit = res
            best_index = i
            # print("best_merit_{}_res_{}".format(best_merit, res))
            # print(best_index)
    # print("best:", best_index)
    # print(a)
    # print(best_index-a)
    # print(F_order_features[best_index-a])
    best_col_name = F_order_features.pop(best_index-a)
    # print(best_col_name)
    F_opt.append(best_col_name)
    return F_opt, F_order_features

def run_cfs(X,y, num_features):
    t1 = time.time()
    F_opt = []
    f_order = get_relevance(X, y) #Get relevancy matrix
    F_order_features = list(f_order.features) #Get feature names in descending order of score
    F_opt.append(F_order_features.pop(0)) 

    bar = progressbar.ProgressBar(maxval=num_features, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    j = 0

    while (len(F_opt) <= num_features):
        F_opt, F_order_features = find_best_feature_parallel(X, y, F_opt, F_order_features)
        bar.update(j+1)
        j+=1
    
    bar.finish()
    walltime = time.time() - t1

    return F_opt, walltime

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
        F_opt, F_order_features = find_best(X, y, F_opt, F_order_features)
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
for path in all_blobs:
    print("Process at: {} %".format((all_blobs.index(path)+1)*100/len(all_blobs)))
    file_name = path.split("/")[-1]
    print(file_name)
    x_train , y_train, x_test, y_test = preprocessing(file_name, False)

    opt, wtime = run_cfs_v3(x_train, y_train, num_features = int(np.ceil(0.3*x_train.shape[1])) )
    with open(dir + path + '_cfs.pkl', 'wb') as f:
        pickle.dump(opt, f)

    with open(dir + path + '_cfs.pkl', 'rb') as f:
        mynewlist = pickle.load(f)

    print(mynewlist)
    print(wtime)

