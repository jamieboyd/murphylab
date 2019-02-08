import sys
import atexit
import platform
import time
import pandas as pd
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

from math import exp, sqrt


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


from pybloqs import Block
import pybloqs.block.table_formatters as tf
from IPython.core.display import display, HTML

filteredtaglist=["201608466","201608468","201608481","201609136","201609336","210608298","2016080026",
                 "2016090793","2016090943",
                 "2016090629","2016090797","2016090882","2016090964","2016090965","2016090985","2016091183",
                 "201608252","201608423","201608474",
                 "801010270","801010219","801010205"]

def getFromDatabase(query):
    db2 = pymysql.connect(host="localhost", user="root", db="murphylab", password='password')
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
    if table == "all":
        query = """SELECT `Mouse`,Time_to_sec(CAST(`Timestamp` AS TIME(0))),DATEDIFF(Date(`Timestamp`),Date("2018-07-24")) AS `Day`,`Last Mouse`,`Time after last Mouse exits` FROM `entries` WHERE 
                ((Date(`entries`.`Timestamp`) BETWEEN "2017-07-12" AND "2017-10-12")
                OR (Date(`entries`.`Timestamp`) BETWEEN "2018-02-14" and "2018-04-01")
                OR (Date(`entries`.`Timestamp`) BETWEEN "2018-04-23" AND "2018-06-01")
                OR (Date(`entries`.`Timestamp`) BETWEEN "2018-07-24" and "2018-10-24")
                OR (Date(`entries`.`Timestamp`) BETWEEN "2018-11-15" AND "2018-12-20"))
                AND (`Trial or Entry`="fix" OR `Trial or Entry`= "nofix") AND `Project_ID` = 3"""
    if table == "stats1":
        query = """SELECT *, `same`/`other` FROM(SELECT `mice_autoheadfix`.`Mouse` as `Mouse`,`mice_autoheadfix`.`cage`, SUM(IF(`entries`.`Mouse`=`Last Mouse`,1,0)) AS `same`,SUM(IF(`entries`.`Mouse`!=`Last Mouse`,1,0)) AS `other` FROM `entries`
                              inner JOIN `mice_autoheadfix` on `mice_autoheadfix`.`Mouse`= `entries`.`Mouse`
                WHERE
                ((Date(`entries`.`Timestamp`) BETWEEN "2017-07-12" AND "2017-10-12")
                OR (Date(`entries`.`Timestamp`) BETWEEN "2018-02-14" and "2018-04-01")
                OR (Date(`entries`.`Timestamp`) BETWEEN "2018-04-23" AND "2018-06-01")
                OR (Date(`entries`.`Timestamp`) BETWEEN "2018-07-24" and "2018-10-24")
                OR (Date(`entries`.`Timestamp`) BETWEEN "2018-11-15" AND "2018-12-20"))
                AND (`Trial or Entry`="fix" OR `Trial or Entry`= "nofix") AND `mice_autoheadfix`.`Activity` = "good"
                GROUP BY `Mouse`
                                                       ORDER BY `mice_autoheadfix`.`cage`)a"""
    elif table == "stats":
        query = """SELECT `Mouse`,Time_to_sec(CAST(`Timestamp` AS TIME(0))),DATEDIFF(Date(`Timestamp`),Date("2018-07-24")) AS `Day`,`Last Mouse`,`Time after last Mouse exits`,
                AVG(IF(`Time after last Mouse exits` < 300 AND `Mouse`=`Last Mouse`,`Time after last Mouse exits`,NULL)) AS `AVG_in_cluster`,
                STDDEV_SAMP(IF(`Time after last Mouse exits` < 300 AND `Mouse`=`Last Mouse`,`Time after last Mouse exits`,NULL)) AS `STD_in_cluster`,
                SUM(IF(`Time after last Mouse exits` < 300 AND `Mouse`=`Last Mouse`,1,NULL)),
                AVG(IF(`Time after last Mouse exits` BETWEEN 300 AND 86400 AND `Mouse`=`Last Mouse`,`Time after last Mouse exits`,NULL)) AS `AVG_between_cluster`,
                STDDEV_SAMP(IF(`Time after last Mouse exits` BETWEEN 300 AND 86400 AND `Mouse`=`Last Mouse`,`Time after last Mouse exits`,NULL)) AS `STD_between_cluster`,
                SUM(IF(`Time after last Mouse exits` BETWEEN 300 AND 86400 AND `Mouse`=`Last Mouse`,1,NULL))
                FROM `entries` WHERE 
                `Mouse` IN ("801010270","801010219","801010205",
                "201608252","201608423","201608474") AND
                ((Date(`entries`.`Timestamp`) BETWEEN "2017-07-12" AND "2017-10-12")
                OR (Date(`entries`.`Timestamp`) BETWEEN "2018-02-14" and "2018-04-01")
                OR (Date(`entries`.`Timestamp`) BETWEEN "2018-04-23" AND "2018-06-01")
                OR (Date(`entries`.`Timestamp`) BETWEEN "2018-07-24" and "2018-10-24")
                OR (Date(`entries`.`Timestamp`) BETWEEN "2018-11-15" AND "2018-12-20"))
                AND (`Trial or Entry`="fix" OR `Trial or Entry`= "nofix") AND (`Project_ID` = 4 OR `Project_ID` = 5)
                GROUP BY `Mouse`
                
                
                `Mouse` IN ("801010270","801010219","801010205",
                "201608252","201608423","201608474") AND"""
    else:
        print("wrong command ")
    return query

def print_entry_times_over_days(df):
    sns.set_style("whitegrid")
    #q = sns.scatterplot(x=df["Time"]/3600,y="Day",data=df, hue="Mouse")
    d = sns.relplot(x=df["Time"] / 3600, y="Day", data=df, hue="Mouse",edgecolor="black")
    sns.despine()
    plt.xlabel("Day time [h]")
    plt.tight_layout()
    plt.show()

def print_entry_histogram(df):
    sns.set_style("whitegrid")
    a = df.loc[(df["Mouse"] != df["Last_mouse"]) &  (df["delta_t_last_mouse"] <= 300),"delta_t_last_mouse"]
    sns.distplot(a=a,rug=False, bins=60, norm_hist=False, kde=False,color= "black", label="different mouse")
    a= df.loc[(df["Mouse"] == df["Last_mouse"]) & (df["delta_t_last_mouse"] <= 300),"delta_t_last_mouse"]
    sns.distplot(a=a,rug=False, bins=60,norm_hist=False, kde=False,color= "red", label="same mouse")
    sns.despine()
    plt.legend()
    plt.xlabel("Seconds after last trial")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()
    a = df.loc[(df["Mouse"] != df["Last_mouse"]) & ((df["delta_t_last_mouse"] >= 300.0 ) &(df["delta_t_last_mouse"] <= 43200 )), "delta_t_last_mouse"]
    sns.distplot(a=a, rug=False, bins=60, norm_hist=False, kde=False, color="black", label="different mouse")
    a = df.loc[(df["Mouse"] == df["Last_mouse"]) & ((df["delta_t_last_mouse"] >= 300.0)&(df["delta_t_last_mouse"] <= 43200 )), "delta_t_last_mouse"]
    sns.distplot(a=a, rug=False, bins=60, norm_hist=False, kde=False, color="red", label="same mouse")
    sns.despine()
    plt.legend()
    plt.xlabel("Seconds after last trial")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()


data = list(getFromDatabase(query = generateQuery("all")))
df = pd.DataFrame(data=data,columns=["Mouse", "Time","Day", "Last_mouse", "delta_t_last_mouse"])
df["Mouse"] = df["Mouse"].str.slice_replace(stop=-3, repl="M")
df["Last_mouse"] = df["Last_mouse"].str.slice_replace(stop=-3, repl="M")
#print_entry_times_over_days(df)
print_entry_histogram(df)

