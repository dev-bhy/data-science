import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("20220629_092015_boruta_walltime.csv") #RF
df = pd.read_csv("20220629_061556_boruta_walltime.csv") #lgboost
print(df.accepted)
newlist = []
numlist = []
for ele in list(df.accepted):
    string1=ele[1:-1]
    lst = list(string1.split(", "))
    if lst == ['']:
        lst == []
        numlist.append(0)
    else:
        numlist.append(len(lst))
    # print(ele)
print(numlist)
print(len(numlist))
df['Num accepted'] = numlist

df.to_csv("rf_with numList.csv")

#Feature size comparisons
# df_plot = pd.DataFrame()
# df_plot['original features'] = df['num input features']
# df_plot['selected features'] = df['Num accepted']

# data_plot =df_plot['selected features'] = df['Num accepted']
# ax = sns.boxplot(data=data_plot, whis=np.inf, palette = ['orange'])
# ax = sns.stripplot( data=data_plot, jitter=0.05, palette=['orange'])
# ax.set_ylabel('Number of features')
# plt.ylim(0, 100)
# # plt.set_ylabel('number of features')
# plt.title('selected features')
# plt.savefig('output_rf_selected.png')

#Walltime comparisons
df_rf = pd.read_csv("20220629_092015_boruta_walltime.csv") #RF
df_lgb = pd.read_csv("20220629_061556_boruta_walltime.csv") #lgboost

df_rf_walltime = df_rf.walltime
df_lgb_walltime = df_lgb.walltime

df_plot2 = pd.DataFrame()

df_plot2['rf runtime'] = df_rf_walltime
df_plot2['lgb runtime'] = df_lgb_walltime

# data_plot =df_plot['selected features'] = df['Num accepted']
ax = sns.boxplot(data=df_plot2, whis=np.inf)
ax = sns.stripplot( data=df_plot2, jitter=0.05)
ax.set_ylabel('walltime (seconds)')
# plt.ylim(0, 200)
# plt.set_ylabel('number of features')
plt.title('walltime')
plt.savefig('walltime.png')