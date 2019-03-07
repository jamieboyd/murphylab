import sys
import atexit
import platform
import time
import pandas as pd
pd.set_option('display.width', 500)
pd.set_option("display.max_columns", 10)
import numpy as np
from sklearn                        import metrics, svm
from sklearn.linear_model           import LinearRegression
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.stats.anova import AnovaRM
import researchpy as rp
from numpy import log
from scipy.stats import norm
import math
from password import database_password as DBpwd

import math
from sklearn import preprocessing
from sklearn import utils
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
import pymysql
import seaborn as sns
from scipy import stats
import six
import glob



filteredtaglist=["201608466","201608468","201608481","201609136","201609336","210608298","2016080026",
                 "2016090793","2016090943",
                 "2016090629","2016090797","2016090882","2016090964","2016090965","2016090985","2016091183",
                 "201608252","201608423","201608474",
                 "801010270","801010219","801010205"]

def getFromDatabase(query):
    db2 = pymysql.connect(host="localhost", user="root", db="murphylab", password=DBpwd)
    cur2 = db2.cursor()
    try:
        cur2.execute(query)
        rows = cur2.fetchall()
    except pymysql.Error as e:
        try:
            print("MySQL Error [%d]: %s" % (e.args[0], e.args[1]))
            return None
        except IndexError:
            print("MySQL Error: %s" % str(e))
            return None
    except TypeError as e:
        print("MySQL Error: TypeError: %s" % str(e))
        return None
    except ValueError as e:
        print("MySQL Error: ValueError: %s" % str(e))
        return None
    db2.close()
    return rows

def generateQuery(table):
    if table == "interactions":
        query = """SELECT `Mouse`,Time_to_sec(CAST(`Timestamp` AS TIME(0))),DATEDIFF(Date(`Timestamp`),
        Date("2017-07-12")) AS `Day`,`Last mouse headfixed`,`Time since last headfix`, UNIX_TIMESTAMP(`Timestamp`),`Project_ID` FROM `entries` WHERE 
                ((Date(`entries`.`Timestamp`) BETWEEN "2017-07-12" AND "2017-10-12")
                OR (Date(`entries`.`Timestamp`) BETWEEN "2018-02-14" and "2018-04-01")
                OR (Date(`entries`.`Timestamp`) BETWEEN "2018-04-23" AND "2018-06-01")
                OR (Date(`entries`.`Timestamp`) BETWEEN "2018-07-24" and "2018-10-24")
                OR (Date(`entries`.`Timestamp`) BETWEEN "2018-11-15" AND "2018-12-20"))
                AND (`Trial or Entry`="fix" OR `Trial or Entry`= "nofix")"""
        #AND `Project_ID` = 4 AND `Mouse` IN ("201608252","201608423","201608474")
    if table == "stats":
        query = """SELECT `quantile`,`gap_threshold`, SUM(IF(`session_count`<= 2,1,0)) AS `few`,SUM(IF(`session_count`>= 3,1,0)) AS `many`,
        COUNT(`session_count`) AS `all`,`cage` FROM `interaction_clusters` GROUP BY `cage`,`quantile`"""
    else:
        print("wrong command ")
    return query

def saveToDatabase(table,vals):
    query, values = generate_commands(table,vals)
    db1 = pymysql.connect(host="localhost",user="root",db="murphylab",password=DBpwd)
    cur1 = db1.cursor()
    try:
        cur1.executemany(query, values)
        db1.commit()
    except pymysql.Error as e:
        try:
            print( "MySQL Error [%d]: %s" % (e.args[0], e.args[1]))
            return None
        except IndexError:
            print( "MySQL Error: %s" % str(e))
            return None
    except TypeError as e:
        print("MySQL Error: TypeError: %s" % str(e))
        return None
    except ValueError as e:
        print("MySQL Error: ValueError: %s" % str(e))
        return None
    db1.close()

def generate_commands(table,vals):
    if table == "interaction_clusters":
        query="""INSERT INTO `interaction_clusters`
        (`quantile`, `gap_threshold`,`session_count`,`mice_count`,`cluster_duration`,`session_time_sum`,`delta_time_sum`,`max_delta`,`gap_length`,`timestamp_start`,`cage`,`day`,`day_or_night`)
            VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,FROM_UNIXTIME(%s),%s,%s,%s)"""
        values = vals
    if table == "interaction_clusters_summary":
        query = """INSERT INTO `interaction_clusters_summary`
                (`Quantile`,`AVG_sessions`,`STD_sessions`,`AVG_mice`,`STD_mice`,`AVG_duration`,`STD_duration`,`AVG_gap`,`STD_gap`,`threshold_gap`,`cage`)
                    VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
        values = vals
    if table == "clusters_kde":
        query = """INSERT INTO `clusters_kde`
        (`Bandwidth`, `Threshold`,`session_count`,`mice_count`,`cluster_duration`,`session_time_sum`,`delta_time_sum`,`max_delta`,`gap_length`,`timestamp_start`,`cage`,`day`,`day_or_night`)
            VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,FROM_UNIXTIME(%s),%s,%s,%s)"""
        values = vals
    return query, values

def print_entry_times_over_days(df):
    sns.set_style("whitegrid")
    sns.set_context("paper",font_scale=2.5)
    fig = plt.figure(figsize=(16,8))
    q = sns.scatterplot(x=df["Time"]/3600,y="Day",data=df, hue="Mouse"
                        ,edgecolor="black",alpha=0.6,s=120)
    #hue_order=["M252", "M423","M242","M474", "M009", "M104", "M008" , "M250"]
    handles, labels = q.get_legend_handles_labels()
    handles = handles[1:]
    labels = labels[1:]
    q.legend(handles=handles, labels=labels, frameon=False, loc=9, ncol=int((len(labels)+1)/2),
             bbox_to_anchor=(0.5, 1.2), markerscale=2)
    sns.despine()
    plt.xlabel("Day time [h]")
    plt.xlim(0, 24)
    plt.tight_layout()
    plt.savefig("bubblegroup.svg", bbox_inches=0, transparent=True)
    plt.show()


def print_entry_histogram(df):
    binning = np.linspace(0,300,61,endpoint=True)
    sns.set(rc={'figure.figsize': (5.5, 4.25)})
    sns.set_style("ticks")
    sns.set_context("paper", font_scale=2.2)

    a = df.loc[(df["Mouse"] != df["Last_mouse"]) &  (df["delta_t_last_mouse"] <= 300),"delta_t_last_mouse"]
    sns.distplot(a=a,rug=False, bins=binning, norm_hist=False, kde=False,color= "black", label="different mouse")
    a= df.loc[(df["Mouse"] == df["Last_mouse"]) & (df["delta_t_last_mouse"] <= 300),"delta_t_last_mouse"]
    sns.distplot(a=a,rug=False, bins=binning,norm_hist=False, kde=False,color= "red", label="same mouse")
    sns.despine()
    plt.legend()
    plt.xlabel("Seconds after last head-fixation")
    plt.ylabel("")
    plt.xlim(0,305)
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("hist5min.svg", bbox_inches=0, transparent=True)
    plt.show()

    binning = np.linspace(100,600,51,endpoint=True) #21600
    a = df.loc[(df["Mouse"] != df["Last_mouse"]) & ((df["delta_t_last_mouse"] >= 100.0 ) &(df["delta_t_last_mouse"] <= 600 )), "delta_t_last_mouse"]
    b = sns.distplot(a=a, rug=False, bins=binning, norm_hist=False, kde=False, color="black", label="different mouse")
    a = df.loc[(df["Mouse"] == df["Last_mouse"]) & ((df["delta_t_last_mouse"] >= 100.0)&(df["delta_t_last_mouse"] <= 600 )), "delta_t_last_mouse"]
    b = sns.distplot(a=a, rug=False, bins=binning, norm_hist=False, kde=False, color="red", label="same mouse")
    sns.despine()
    plt.legend()
    plt.xlabel("Minutes after last head-fixation")
    plt.ylabel("")
    plt.xticks([120, 180,240,300,360,420,480,540,600], [2, 3, 4, 5, 6, 7, 8,9,10])
    plt.xlim(100,600)
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("hist10min.svg", bbox_inches=0, transparent=True)
    plt.show()

    binning = np.linspace(0,21600, 73, endpoint=True)  # 21600
    a = df.loc[(df["Mouse"] != df["Last_mouse"]) & (
                (df["delta_t_last_mouse"] >= 00.0) & (df["delta_t_last_mouse"] <= 21600)), "delta_t_last_mouse"]
    b = sns.distplot(a=a, rug=False, bins=binning, norm_hist=False, kde=False, color="black", label="different mouse")
    a = df.loc[(df["Mouse"] == df["Last_mouse"]) & (
                (df["delta_t_last_mouse"] >= 00.0) & (df["delta_t_last_mouse"] <= 21600)), "delta_t_last_mouse"]
    b = sns.distplot(a=a, rug=False, bins=binning, norm_hist=False, kde=False, color="red", label="same mouse")
    sns.despine()
    plt.xticks([0,3600,7200,10800,14400,18000,21600], [0,1,2,3,4,5,6])
    plt.xlim(0,21600)
    plt.legend()
    plt.xlabel("Hours after last head-fixation")
    plt.ylabel("")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("hist6hours.svg", bbox_inches=0, transparent=True)
    plt.show()

def day_night(row):
    lighton = 7
    lightoff = 19
    if (row["Time"] >= lighton*3600 and row["Time"] <= lightoff*3600):
        val = "day"
    else:
        val = "night"
    return val

def clustering(df,QUANTILE,cage):
    df = df.sort_values(by=["Day","Time"])
    df["day_night"] = df.apply(day_night, axis=1)                                          #classify data by night and day
    quantile_thresh = df["delta_t_last_mouse"].quantile(QUANTILE/100)
    array = df.values.tolist()

    Cluster_list = []
    mouse_list = []
    delta_list = [0]
    COUNT_SESSIONS = 0
    SESSION_SUM = 0
    for i in range(len(array)):
        MOUSE = array[i][0]
        time = int(array[i][1])
        DAY = int(array[i][2])
        DELTA = array[i][4]
        UNIX = int(array[i][5])
        DAY_NIGHT = str(array[i][6])
        if i == 0:
            first_timestamp = UNIX
        if DELTA >= quantile_thresh:
            if i > 0:
                COUNT_MICE = len(mouse_list)
                DURATION_CLUSTER = round((UNIX - first_timestamp - DELTA), 2)
                DELTA_SUM = sum(delta_list)
                MAX_DELTA = max(delta_list)
                SESSION_SUM = DURATION_CLUSTER - DELTA_SUM
                row_list = [QUANTILE,quantile_thresh,COUNT_SESSIONS, COUNT_MICE, DURATION_CLUSTER, SESSION_SUM, DELTA_SUM, MAX_DELTA,
                            DELTA,  first_timestamp, cage, DAY,DAY_NIGHT]
                Cluster_list.append(row_list)
            # reset cluster, get ready for new one
            COUNT_SESSIONS = 0
            SESSION_SUM = 0
            delta_list = [0]
            mouse_list = []
            first_timestamp = UNIX
        if MOUSE not in mouse_list:
            mouse_list.append(MOUSE)
        if COUNT_SESSIONS > 0:
            delta_list.append(DELTA)
        COUNT_SESSIONS += 1
    saveToDatabase("interaction_clusters", Cluster_list)
    df2 = pd.DataFrame(data=Cluster_list,
                       columns=["quantile", "threshold", "cluster_sessions", "cluster_mice", "cluster_length","session_time_sum","delta_time_sum","max_delta",
                                "cluster_gap", "cluster_starttime", "cage", "day","day_or_night"])

    summary = [df2["quantile"].mean().item(), df2["cluster_sessions"].mean().item(),
               df2["cluster_sessions"].std().item(),
               df2["cluster_mice"].mean().item(), df2["cluster_mice"].std().item(),
               df2["cluster_length"].mean().item(),
               df2["cluster_length"].std().item(), df2["cluster_gap"].mean().item(),
               df2["cluster_gap"].std().item(),
               df2["threshold"].mean().item(), df2["cage"].max()]
    return summary
def save_clusters(df,cage):
    summary_list = []
    for i in range(50, 100):
        summary = clustering(df, i, cage)   #make sure the query corresponds with the statement! "all"-statement is saved in the DB, change to a number when using distinct cage and also include the selection in the "all"-query
        summary_list.append(summary)
    saveToDatabase("interaction_clusters_summary", summary_list)

def quantile_verification():
    query = """SELECT `quantile`,`gap_threshold`, SUM(IF(`session_count`<= 3,1,0)) AS `few`,SUM(IF(`session_count`>= 4,1,0)) AS `many`,
            COUNT(`session_count`) AS `all`,`cage` FROM `interaction_clusters` GROUP BY `cage`,`quantile`"""
    data = list(getFromDatabase(query=query))
    df = pd.DataFrame(data=data, columns=["quantile","gap_threshold", "count_few_sessions", "count_many_sessions", "count_all_sessions", "cage"])
    df["ratio"] = (df["count_few_sessions"]*df["count_many_sessions"])/(df["count_all_sessions"])**2

    a = sns.lineplot(data=df,x="quantile",y="ratio",hue="cage",estimator=None,legend=False)
    b = a.twinx()
    sns.lineplot(data=df, x="quantile", y="gap_threshold", hue="cage", estimator=None)
    b.set_yscale("log")
    plt.show()

    c = sns.lineplot(data=df,x="gap_threshold",y="ratio",hue="cage",estimator=None)
    c.set_xscale("log")
    plt.show()

    df["ratio"] = (df["count_few_sessions"] * df["count_many_sessions"]) / (df["count_all_sessions"]) ** 2
    d = sns.lineplot(data=df,x="quantile",y="ratio",hue="cage",estimator=None)


def kerneldensity(df, cage):
    from numpy import array, linspace
    from sklearn.neighbors.kde import KernelDensity
    from matplotlib.pyplot import plot
    from scipy.signal import argrelextrema
    #densities
    BANDWIDTH=50
    df = df.sort_values(by=["Unix"])
    timestamps_headfixes = array(df["Unix"])[:, np.newaxis]
    kde = KernelDensity(kernel='gaussian', bandwidth=BANDWIDTH).fit(timestamps_headfixes)
    target_x_values = linspace(int(min(timestamps_headfixes)),int(max(timestamps_headfixes))+2,int(max(timestamps_headfixes))-int(min(timestamps_headfixes))+2)[:, np.newaxis]
    densities = np.exp(kde.score_samples(target_x_values))
    densities_log = kde.score_samples(target_x_values)
    #and minima
    minima_indices = argrelextrema(densities_log, np.less, order=1)[0]
    threshold = 0.09*10**(-7)
    mini = [i for i in minima_indices if densities[i] <=threshold]
    mini.append(int(max(timestamps_headfixes))-int(min(timestamps_headfixes))-1)
    timestamp_borders = target_x_values[mini]
    #timestamp_borders = np.append(timestamp_borders,array([max(timestamps_headfixes)+1]),axis=0)
    print(timestamp_borders)
    array = df.values.tolist()
    Cluster_list = []
    mouse_list = []
    delta_list = [0]
    COUNT_SESSIONS = 0
    minima_counter = 0
    SESSION_SUM = 0
    for i in range(len(array)):
        next_cluster_threshold = timestamp_borders[minima_counter]
        MOUSE = array[i][0]
        time = int(array[i][1])
        DAY = int(array[i][2])
        DELTA = array[i][4]
        UNIX = int(array[i][5])
        DAY_NIGHT = str(array[i][6])
        if i == 0:
            first_timestamp = UNIX
        if UNIX >= next_cluster_threshold:
            minima_counter +=1
            if i > 0:
                COUNT_MICE = len(mouse_list)
                DURATION_CLUSTER = round((UNIX - first_timestamp - DELTA), 2)
                DELTA_SUM = sum(delta_list)
                MAX_DELTA = max(delta_list)
                SESSION_SUM = DURATION_CLUSTER - DELTA_SUM
                row_list = [BANDWIDTH,threshold,COUNT_SESSIONS, COUNT_MICE, DURATION_CLUSTER, SESSION_SUM, DELTA_SUM, MAX_DELTA,
                            DELTA,  first_timestamp, cage, DAY,DAY_NIGHT]
                Cluster_list.append(row_list)
            # reset cluster, get ready for new one
            COUNT_SESSIONS = 0
            SESSION_SUM = 0
            delta_list = [0]
            mouse_list = []
            first_timestamp = UNIX
        if MOUSE not in mouse_list:
            mouse_list.append(MOUSE)
        if COUNT_SESSIONS > 0:
            delta_list.append(DELTA)
        COUNT_SESSIONS += 1

    #saveToDatabase("clusters_kde", Cluster_list)

    #for i in range(len(mini)-1):
    #    print(len(timestamps_headfixes[(timestamps_headfixes >= target_x_values[mini[i]]) & (timestamps_headfixes <= target_x_values[mini[i+1]])]))
    plot(target_x_values, densities)
    plt.show()

def analyze_clusters():
    bins = [1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 75, 100, 1000]
    labels = [1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 75, 100]
    labls = [0, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 75, 100]
    sns.set_context("paper", font_scale=1.75)
    query= """SELECT `session_count`, COUNT(`session_count`), SUM(`session_count`),AVG(`cluster_duration`),
     AVG(`cluster_duration`)/`session_count`FROM `clusters_kde`
GROUP BY `session_count`"""
    data = list(getFromDatabase(query=query))
    df = pd.DataFrame(data=data,
                      columns=["cluster size", "_session_count", "headfixes","Avg cluster duration","Avg duration per session"])
    df = df.groupby([pd.cut(df['cluster size'], bins=bins, labels=labels,right=False)])["headfixes"].sum().reset_index()
    bar = sns.barplot(x="cluster size",data=df,y="headfixes")

    point = bar.twinx()
    data1 = list(getFromDatabase(query="""SELECT `gap_length`/`delta_time_sum`,`session_count`,`delta_time_sum`/(`session_count`-1) FROM `clusters_kde`"""))
    df1 = pd.DataFrame(data=data1, columns=["time ratio between and sum within clusters","cluster size","ratio to avg delta"])
    df1["cut"] = pd.cut(df1['cluster size'], bins=bins, labels=labels,right=False)
    print(df1)
    point = sns.pointplot(data=df1,x="cut",y="time ratio between and sum within clusters",color="dimgrey",capsize=0.2 ,ci=95)
    point2 =sns.pointplot(data=df1, x="cut", y="ratio to avg delta", color="black", capsize=0.2, ci="sd")
    point.set_yscale("log")
    sns.despine(top=True,right=False)
    plt.show()

def day_night_mice_count():
    sns.set_context("paper", font_scale=1.75)
    data = list(getFromDatabase(query="""SELECT SUM(IF(`mice_count`=1,1,0))/count(*)*100 AS `1`, SUM(IF(`mice_count`=2,1,0))/count(*)*100 AS `2`,
         SUM(IF(`mice_count`>2,1,0))/count(*)*100 AS `more`,`cage`,count(*) FROM `clusters_kde` GROUP BY `cage`"""))
    df_involvedmice = pd.DataFrame(data=data, columns=["one", "two", "more", "Group", "count"])
    df_involvedmice = pd.melt(df_involvedmice, id_vars=["Group", "count"], value_vars=["one", "two", "more"],
                              var_name='Mice in cluster', value_name='Distribution [%]')

    a = sns.barplot(data=df_involvedmice, x="Group", y='Distribution [%]', hue='Mice in cluster', dodge=False,
                    saturation=1, palette="PuBuGn_d")
    data = list(getFromDatabase(
        query="""SELECT SUM(IF(`day_or_night`="day",`session_count`,0))/ SUM(`session_count`)*100 as `Day_night_ratio`,`cage` FROM `clusters_kde` GROUP BY `cage`"""))
    df_day_night = pd.DataFrame(data=data, columns=["light period activity [%]", "Group"])
    b = a.twinx()
    a.set_ylim(0, 100)
    b.set_ylim(0, 100)
    b = sns.lineplot(x="Group", y="light period activity [%]", data=df_day_night, estimator=None, color="silver",
                     linewidth = 4)
    a.legend(frameon=False, bbox_to_anchor=(0.5, 1.2),title="Mice in cluster",ncol=3,loc=9)
    sns.despine(top=True, right=False, bottom=True)

    plt.tight_layout()
    plt.show()

def tests(df):
    df = df.sort_values(by=["delta_t_last_mouse"])
    ander = stats.chisquare(df["delta_t_last_mouse"],ddof=0)
    print(ander)
def querys():
    #ratio of sessions did alone and sessions did as group
    query = """SELECT SUM(IF(`mice_count`=1,1,0))/count(*) AS `1`, SUM(IF(`mice_count`=2,1,0))/count(*) AS `2`,
     SUM(IF(`mice_count`>2,1,0))/count(*) AS `more`,`cage`,count(*) FROM `clusters_kde` GROUP BY `cage`"""
    query = """SELECT AVG(`session_count`), `mice_count`,count(*), AVG(`session_count`)/ `mice_count` FROM `clusters_kde`
     WHERE `session_count` > `mice_count`*4 GROUP BY `mice_count`"""
    query = """SELECT Date(`timestamp_start`) as `date`, SUM(`session_count`), count(`session_count`),AVG(`delta_time_sum`/(`session_count`-1)),
     SUM(`gap_length`),SUM(`cluster_duration`),SUM(`session_time_sum`),SUM(`session_time_sum`)/max(`mice_count`)  FROM `clusters_kde`
GROUP BY `date`"""

data = list(getFromDatabase(query = generateQuery("interactions")))
df = pd.DataFrame(data=data,columns=["Mouse", "Time","Day", "Last_mouse", "delta_t_last_mouse","Unix","Group"])
df = df.sort_values(by=["Unix"])
df["day_night"] = df.apply(day_night, axis=1)
medians=[
df.loc[((df["Mouse"] == df["Last_mouse"]) & (df["Group"] == 1)),"delta_t_last_mouse"].median(),
df.loc[((df["Mouse"] != df["Last_mouse"]) & (df["Group"] == 1)),"delta_t_last_mouse"].median(),
df.loc[((df["Mouse"] == df["Last_mouse"]) & (df["Group"] == 2)),"delta_t_last_mouse"].median(),
df.loc[((df["Mouse"] != df["Last_mouse"]) & (df["Group"] == 2)),"delta_t_last_mouse"].median(),
df.loc[((df["Mouse"] == df["Last_mouse"]) & (df["Group"] == 3)),"delta_t_last_mouse"].median(),
df.loc[((df["Mouse"] != df["Last_mouse"]) & (df["Group"] == 3)),"delta_t_last_mouse"].median(),
df.loc[((df["Mouse"] == df["Last_mouse"]) & (df["Group"] == 4)),"delta_t_last_mouse"].median(),
df.loc[((df["Mouse"] != df["Last_mouse"]) & (df["Group"] == 4)),"delta_t_last_mouse"].median(),
df.loc[((df["Mouse"] == df["Last_mouse"]) & (df["Group"] == 5)),"delta_t_last_mouse"].median(),
df.loc[((df["Mouse"] != df["Last_mouse"]) & (df["Group"] == 5)),"delta_t_last_mouse"].median(),
]
print(medians)
#save_clusters(df, cage="all")
#kerneldensity(df, cage="5")

df["Mouse"] = df["Mouse"].str.slice_replace(stop=-3, repl="M")
df["Last_mouse"] = df["Last_mouse"].str.slice_replace(stop=-3, repl="M")
#quantile_verification()
day_night_mice_count()
#analyze_clusters()
#print_entry_times_over_days(df)
#print_entry_histogram(df)
tests(df)



