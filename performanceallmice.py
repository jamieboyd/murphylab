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
def generateQuery(detailed):
    if detailed == "Yes":
        query = """SELECT t.`ts1`,t.`Mouse`,t. `Fixation`, t.`Task`,t.`Outcome`,
                    t.`Trials`,s.`Sumtrials`as 'Trials this day' ,ROUND(t.`Trials`/Sum(s.`Sumtrials`)*100,2) as '% Performance',t.`Project_ID` FROM
                    (select
                    Date(`Trial start`) as `ts1`,
                    `Mouse`,count(*) as `Trials`,`Task`,`Fixation`,`Outcome`,`Project_ID`
                    from `headfix_trials_summary` where
                    ((Date(`Trial start`) BETWEEN "2017-08-28" and "2017-10-12") OR 
                    (Date(`Trial start`) BETWEEN "2018-02-20" and "2018-04-09") OR 
                    (Date(`Trial start`) BETWEEN "2018-04-28" and "2018-06-01") OR 
                    (Date(`Trial start`) BETWEEN "2018-08-08" and "2018-10-23") OR 
                    (Date(`Trial start`) BETWEEN "2018-11-23" AND "2018-12-20"))
                      AND `Trial in session` < 10
                    group by `ts1`, `Mouse`,`Notes`, `Fixation`) t
                    INNER JOIN (SELECT `ts1`,`Mouse`, `Fixation`,`Task`,Sum(`Trials`) as `Sumtrials` FROM
                    (select
                    Date(`Trial start`) as `ts1`,`Mouse`,count(*) as `Trials`,`Task`,`Fixation`,`Outcome`
                    from `headfix_trials_summary` where
                    ((Date(`Trial start`) BETWEEN "2017-08-28" and "2017-10-12") OR 
                    (Date(`Trial start`) BETWEEN "2018-02-20" and "2018-04-09") OR 
                    (Date(`Trial start`) BETWEEN "2018-04-28" and "2018-06-01") OR 
                    (Date(`Trial start`) BETWEEN "2018-08-08" and "2018-10-23") OR 
                    (Date(`Trial start`) BETWEEN "2018-11-23" AND "2018-12-20"))
                     AND `Trial in session` < 10
                    group by `ts1`, `Mouse`,`Outcome`, `Fixation`) k
                    GROUP by `ts1`,`Mouse`, `Fixation`,`Task`
                    ORDER by `ts1`,`Mouse`,`Task`) s ON s.`Mouse` = t.`Mouse` AND s.`ts1` = t.`ts1` AND s.`Fixation` = t.`Fixation` AND s.`Task` = t.`Task`
                    GROUP by `ts1`,`Mouse`, `Fixation`,`Outcome`
                    ORDER by `ts1`,`Mouse`, `Fixation`,`Task`,`Outcome`"""
    elif detailed == "No":
        query = """SELECT t.`ts1`,t.`Mouse`,t. `Fixation`, t.`Task`,t.`Notes`,
                    t.`Trials`,s.`Sumtrials`as 'Trials this day' ,ROUND(t.`Trials`/Sum(s.`Sumtrials`)*100,2) as `% Performance`,t.`Project_ID` FROM
                    (select
                    Date(`Trial start`) as `ts1`,
                    `Mouse`,count(*) as `Trials`,`Task`,`Fixation`,`Notes`,`Project_ID`
                    from `headfix_trials_summary` where
                    ((Date(`Trial start`) BETWEEN "2017-08-28" and "2017-10-12") OR 
                    (Date(`Trial start`) BETWEEN "2018-02-20" and "2018-04-09") OR 
                    (Date(`Trial start`) BETWEEN "2018-04-28" and "2018-06-01") OR 
                    (Date(`Trial start`) BETWEEN "2018-08-08" and "2018-10-23") OR 
                    (Date(`Trial start`) BETWEEN "2018-11-23" AND "2018-12-20"))
                      AND `Trial in session` < 10 AND Date(`Trial start`) != "2018-05-12"
                      AND Date(`Trial start`) != "2017-09-07" AND Date(`Trial start`) != "2018-05-22" AND Date(`Trial start`) != "2018-05-21"
                    group by `ts1`, `Mouse`,`Notes`, `Fixation`) t
                    INNER JOIN (SELECT `ts1`,`Mouse`, `Fixation`,`Task`,Sum(`Trials`) as `Sumtrials` FROM
                    (select
                    Date(`Trial start`) as `ts1`,`Mouse`,count(*) as `Trials`,`Task`,`Fixation`,`Notes`
                    from `headfix_trials_summary` where
                    ((Date(`Trial start`) BETWEEN "2017-08-28" and "2017-10-12") OR 
                    (Date(`Trial start`) BETWEEN "2018-02-20" and "2018-04-09") OR 
                    (Date(`Trial start`) BETWEEN "2018-04-28" and "2018-06-01") OR 
                    (Date(`Trial start`) BETWEEN "2018-08-08" and "2018-10-23") OR 
                    (Date(`Trial start`) BETWEEN "2018-11-23" AND "2018-12-20"))
                     AND `Trial in session` < 10 AND Date(`Trial start`) != "2018-05-12"
                     AND Date(`Trial start`) != "2017-09-07" AND Date(`Trial start`) != "2018-05-22" AND Date(`Trial start`) != "2018-05-21"
                    group by `ts1`, `Mouse`,`Notes`, `Fixation`) k
                    GROUP by `ts1`,`Mouse`, `Fixation`,`Task`
                    ORDER by `ts1`,`Mouse`,`Task`) s ON s.`Mouse` = t.`Mouse` AND s.`ts1` = t.`ts1` AND s.`Fixation` = t.`Fixation` AND s.`Task` = t.`Task`
                    where s.`Sumtrials` > 1 
                    GROUP by `ts1`,`Mouse`, `Fixation`,`Notes`
                    ORDER by `ts1`,`Mouse`, `Fixation`,`Task`,`Notes`"""
    elif detailed == "pooled":
        query = """SELECT u.`Mouse`,u.`Fixation`, u.`Task`,u.`Notes`,
                    SUM(u.`Trials`),SUM(u.`Trials this day`),
                    ROUND((SUM(u.`Trials`)/SUM(u.`Trials this day`))*100,1) AS `ratio`, ROUND(AVG(u.`% Performance`),1),ROUND(STDDEV(u.`% Performance`),1), u.`Project_ID`
                    FROM
                    (SELECT t.`ts1`,t.`Mouse`,t. `Fixation`, t.`Task`,t.`Notes`,
                    t.`Trials`,s.`Sumtrials`as `Trials this day` ,ROUND(t.`Trials`/Sum(s.`Sumtrials`)*100,2) as `% Performance`,t.`Project_ID` FROM
                    (select
                    Date(`Trial start`) as `ts1`,
                    `Mouse`,count(*) as `Trials`,`Task`,`Fixation`,`Notes`,`Project_ID`
                    from `headfix_trials_summary` where
                    ((Date(`Trial start`) BETWEEN "2017-08-28" and "2017-10-12") OR 
                    (Date(`Trial start`) BETWEEN "2018-02-20" and "2018-04-09") OR 
                    (Date(`Trial start`) BETWEEN "2018-04-28" and "2018-06-01") OR 
                    (Date(`Trial start`) BETWEEN "2018-08-08" and "2018-10-23") OR 
                    (Date(`Trial start`) BETWEEN "2018-11-23" AND "2018-12-20"))
                      AND `Trial in session` < 10 AND Date(`Trial start`) != "2018-05-12"
                      AND Date(`Trial start`) != "2017-09-07" AND Date(`Trial start`) != "2018-05-22" AND Date(`Trial start`) != "2018-05-21"
                    group by `ts1`, `Mouse`,`Notes`, `Fixation`) t
                    INNER JOIN (SELECT `ts1`,`Mouse`, `Fixation`,`Task`,Sum(`Trials`) as `Sumtrials` FROM
                    (select
                    Date(`Trial start`) as `ts1`,`Mouse`,count(*) as `Trials`,`Task`,`Fixation`,`Notes`
                    from `headfix_trials_summary` where
                    ((Date(`Trial start`) BETWEEN "2017-08-28" and "2017-10-12") OR 
                    (Date(`Trial start`) BETWEEN "2018-02-20" and "2018-03-09") OR 
                    (Date(`Trial start`) BETWEEN "2018-04-28" and "2018-06-01") OR 
                    (Date(`Trial start`) BETWEEN "2018-08-08" and "2018-10-23") OR 
                    (Date(`Trial start`) BETWEEN "2018-11-23" AND "2018-12-20"))
                     AND `Trial in session` < 10 AND Date(`Trial start`) != "2018-05-12"
                     AND Date(`Trial start`) != "2017-09-07" AND Date(`Trial start`) != "2018-05-22" AND Date(`Trial start`) != "2018-05-21"
                    group by `ts1`, `Mouse`,`Notes`, `Fixation`) k
                    GROUP by `ts1`,`Mouse`, `Fixation`,`Task`
                    ORDER by `ts1`,`Mouse`,`Task`) s ON s.`Mouse` = t.`Mouse` AND s.`ts1` = t.`ts1` AND s.`Fixation` = t.`Fixation` AND s.`Task` = t.`Task`
                    where s.`Sumtrials` > 1 
                    GROUP by `ts1`,`Mouse`, `Fixation`,`Notes`
                    ORDER by `ts1`,`Mouse`, `Fixation`,`Task`,`Notes`)u
                    WHERE u.`Fixation` = "fix" AND u.`Notes` = "GO=2"
                    GROUP BY u.`Mouse`, u.`Fixation`,u.`Notes`"""
    elif detailed == "dPrime":
        query = """SELECT DATEDIFF(Date(`Trial start`),Date("2018-08-08")) AS `Day`,`Mouse`,
                    SUM(IF(`Notes` = "GO=-4" OR `Notes` = "GO=2" ,1,NULL)) AS `HIT`, SUM(IF(`Notes` = "GO=-2",1,NULL)) AS `MISS`,
                    SUM(IF(`Notes` = "GO=-1" OR `Notes` = "GO=-3" ,1,NULL)) AS `FA`, SUM(IF(`Notes` = "GO=1" ,1,NULL)) AS `CR`
                    FROM `headfix_trials_summary`
                    WHERE `Project_ID` = 4 AND Date(`Trial start`) >= "2018-08-21" AND `Fixation` = "fix"
                    GROUP BY `Day`,`Mouse`"""
    else:
        print("wrong command ")
    return query
def save_table(table,dataframe):
    if table == "together":
        df3 = dataframe.groupby(["Day", "Fixation"])["% Performance"].mean()
        df3 = pd.DataFrame(df3)
        df3.to_html("allmiceperformance.html", col_space=80)
    elif table == "counts":
        df3 = dataframe.groupby(["Day", "Fixation"])["Mouse"].count()
        df3 = pd.DataFrame(df3)
        df3.to_html("dayscounts.html", col_space=80)
def prepare_dataframes(df,detailed,list):
    if detailed == "No":
        df.replace({"Outcome": {"GO=2": "GO", "GO=-2": "fail: no licks", "GO=-4": "fail: licked before stimulus ended",
                                "GO=1": "NO GO", "GO=-1": "fail: licked, but right time window",
                                "GO=-3": "fail:licked and wrong time window"}}, inplace=True)
    elif detailed == "Yes":
        df.replace({"Outcome": {"correct": "GO", "correct: is active": "NO GO"}}, inplace=True)
    df.replace({"Cage": {1: "2017-08-28", 2: "2018-02-20", 3: "2018-04-28", 4: "2018-08-08", 5: "2018-11-23"}}, inplace=True)

    df["% Performance"] = df["% Performance"].convert_objects(convert_numeric=True)
    df['Date'] = pd.to_datetime(df['Date'])
    #df['Cage'] = pd.to_datetime(df['Cage'])
    df["Day"] = (df["Date"] - pd.to_datetime(df['Cage'])).dt.days
    df.replace({"Cage": {"2017-08-28": "Group 1", "2018-02-20":"Group 2","2018-04-28":"Group 3", "2018-08-08":"Group 4", "2018-11-23":"Group 5"}}, inplace=True)

    df1 = df[df["Mouse"].isin(cage4taglist)]
    df1 = df1[df1["Outcome"].isin(outcomelist)]

    df2 = df[df["Mouse"].isin(filteredtaglist)]
    df2 = df2[df2["Outcome"].isin(list)]
    df2 = df2[df2.Day <= 60]
    return df1,df2

def linear_fit(X,y):
    X = sm.add_constant(X, prepend=False)
    model = sm.OLS(y, X)
    result = model.fit()
    print(result.summary())
    return result

def TWOway_anovaRM(df):
    doublefilteredtaglist = ["2016090793", "2016090943",
                             "2016090629", "2016090797", "2016090965",
                             "201608252", "201608423",
                             "801010219"] #, "801010270"
    df1 = df[df["Mouse"].isin(doublefilteredtaglist)]
    df1 = df1[df1.Day <= 30]
    binnings = [-1,4,9,14,19,24,30]
    binlabels = [5,10,15,20,25,30]

    df_groupby_fixation_binned = df1.groupby([pd.cut(df1["Day"],bins=binnings,labels=binlabels ), "Mouse", "Fixation"])["% Performance"].mean().reset_index()
    df_groupby_fixation_binned.to_html("binned.html", bold_rows=False, col_space=80)
    anova = AnovaRM(data = df_groupby_fixation_binned,depvar="% Performance",subject="Mouse", within=["Day","Fixation"])
    result = anova.fit()
    print(result.summary)
    #text = result.
    print(result)
    fig = plt.figure(figsize=(7, 5))
    sns.set(style="ticks",font_scale=2,context="paper")
    anovaplot = sns.pointplot(x= "Day", y= "% Performance", hue= "Fixation", capsize=.2, palette="YlGnBu_d", height=6, aspect=.95,
                kind="point", data=df_groupby_fixation_binned,legend_out=True,markers='o')

    handles, _ = anovaplot.get_legend_handles_labels()
    anovaplot.legend(handles= handles,labels=["Headfixed","Not fixed"], frameon=False, loc=9, ncol=2, bbox_to_anchor=(0.5, 1.2),markerscale=2)
    plt.ylim(0, 80)
    plt.tight_layout()
    sns.despine()
    plt.setp(anovaplot.collections, sizes=[120])
    plt.xlabel("")
    plt.ylabel("")
    plt.savefig("anova.svg", bbox_inches=0, transparent=True)
    plt.show()
    return result




Z = norm.ppf
def dPrime(hits, misses, fas, crs):
    # Floors an ceilings are replaced by half hits and half FA's
    halfHit = 0.5/(hits + misses)
    halfFa = 0.5/(fas + crs)

    # Calculate hitrate and avoid d' infinity
    hitRate = hits/(hits + misses)
    if hitRate == 1: hitRate = 1 - halfHit
    if hitRate == 0: hitRate = halfHit

    # Calculate false alarm rate and avoid d' infinity
    faRate = fas / (fas + crs)
    if faRate == 1: faRate = 1 - halfFa
    if faRate == 0: faRate = halfFa

    # Return d', beta, c and Ad'
    #out = {}
    #out['d'] = Z(hitRate) - Z(faRate)
    #out['beta'] = exp((Z(faRate) ** 2 - Z(hitRate) ** 2) / 2)
    #out['c'] = -(Z(hitRate) + Z(faRate)) / 2
    #out['Ad'] = norm.cdf(out['d'] / sqrt(2))
    out = Z(hitRate) - Z(faRate)
    return out

def dPrime_dataframe():
    data = list(getFromDatabase(generateQuery("dPrime")))
    df = pd.DataFrame(data=data,columns=["Day","Mouse","HIT","MISS","FA","CR"])
    df.fillna(value=0, inplace=True)
    dprime=[]
    for index, row in df.iterrows():
        hit = float(row["HIT"])
        miss = float(row["MISS"])
        fa = float(row["FA"])
        cr = float(row["CR"])
        dprime.append( float(dPrime(hit,miss,fa,cr)))
    df["d'"] = dprime
    print(df)
    dprime_plot = sns.lineplot(x="Day",y="d'",data=df,hue="Mouse",palette=["r","b","g"])
    sns.despine()
    plt.tight_layout()
    plt.show()
def make_augmented_dickey_fuller_test(y, trend):
    y = y
    z = log(y)
    result = adfuller(x=y, regression=trend)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

    result = adfuller(x=z, regression=trend)
    print('ADF Statistic log-scaled: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))




def draw_singleplot(df2,hue_value,style_value):
    hue_value = hue_value
    sns.set_context("poster")
    f, axes = plt.subplots(1, 1, figsize=(16, 8))

    sns.set_style("ticks", {"xtick.major.size": 4, "ytick.major.size": 4})
    sns.despine()

    p = sns.countplot(data=df2, x="Day", hue="Cage")
    f = sns.lineplot(data=df2, y="% Performance", x="Day", hue=hue_value,style=style_value , estimator="mean", ci=90,
                     legend=False).set(ylabel="Success rate [%] \n (n mice) ")
    p.legend(frameon=False, loc=9, ncol=2)
    for ind, label in enumerate(p.get_xticklabels()):
        if ind % 10 == 0:  # every 10th label is kept
            label.set_visible(True)
        else:
            label.set_visible(False)

    # //////////////////// regression
    X = df_groupby_fixation["Day"].values.reshape(-1, 1)
    y = df_groupby_fixation.iloc[:, 1].values
    regression = linear_fit(X,y)
    slope = regression.params[0]
    intercept = regression.params[1]
    score = regression.rsquared
    slope_error = regression.bse[0]
    intercept_error = regression.bse[1]
    X_plot = np.linspace(0, 85, 100)
    Y_plot = intercept + X_plot * slope
    #plt.plot(X_plot, Y_plot, color='r')
    text = "linear fit [R$^2$={}]: \n SR(d) = {}($\pm${}) + d $\cdot$ {}($\pm${})  "\
        .format(score.round(2),intercept.round(1),intercept_error.round(1),slope.round(2),slope_error.round(2))
    #plt.text(30, 75, text, horizontalalignment='center', size='small', color='black')
    #//////////////////////////////////////////////////////////////////////////

    plt.xlim(-1, 60)
    plt.ylim(0, 100)
    p.set_title("Pooled statistics of GO trials")
    sns.despine()
    plt.tight_layout()
    plt.show()
def draw_doubleplot(df1,df2,hue_value):
    hue_value = hue_value
    df1.rename(columns={"Outcome": "Task"}, inplace=True)

    sns.set_context("poster")
    f, axes = plt.subplots(2, 1, figsize=(16, 16), sharex=True)

    sns.set_style("ticks", {"xtick.major.size": 4, "ytick.major.size": 4})
    sns.despine()

    p = sns.countplot(data=df2, x="Day", hue="Fixation", ax=axes[0])
    f = sns.lineplot(data=df2, y="% Performance", x="Day", hue=hue_value, size=hue_value, estimator="mean", ci=90,
                     legend=False, ax=axes[0]).set(ylabel="Success rate [%] \n (n mice) ")
    p.legend(frameon=False, loc=9, ncol=2)
    for ind, label in enumerate(p.get_xticklabels()):
        if ind % 10 == 0:  # every 10th label is kept
            label.set_visible(True)
        else:
            label.set_visible(False)

    plt.xlim(-1, 60)
    plt.ylim(0, 100)
    p.set_title("Pooled statistics of GO trials")

    # //////////////////////////////////////////////////////////////////////

    a = sns.lineplot(data=df1, y="% Performance", x="Day", hue="Task", style="Fixation", size="Fixation", ax=axes[1])
    handles, labels = a.get_legend_handles_labels()
    handles = handles[1:3] + handles[4:6]
    labels = labels[1:3] + labels[4:6]
    print(handles, labels)
    a.legend(handles=handles, labels=labels, frameon=False, loc=9, ncol=4, bbox_to_anchor=(0.5, 1.2),
             title="Mouse 201608423")
    a.set(ylabel="Success rate [%] \n ")
    for ind, label in enumerate(a.get_xticklabels()):
        if ind % 10 == 0:  # every 10th label is kept
            label.set_visible(True)
        else:
            label.set_visible(False)

    plt.xlim(-1, 60)
    plt.ylim(0, 100)
    sns.despine()

    sns.despine()
    plt.tight_layout()
    plt.show()
def draw_countplot(df):
    df1 = df.round({"% Performance": -1})
    sns.set_context("talk")
    f, axes = plt.subplots(1, 1, figsize=(16, 8))

    sns.set_style("ticks", {"xtick.major.size": 4, "ytick.major.size": 4})
    sns.despine()

    p = sns.countplot(data=df1, x="% Performance", hue="Fixation").set(ylabel="count Success rate [%] ")
    plt.tight_layout()
    plt.show()
def draw_histogram(df):
    sns.set_style("white")
    sns.set_context("talk")
    #g = sns.countplot(data=df, x="% Performance", hue="Fixation")
    p = sns.distplot(df.loc[df["Fixation"] == "fix","% Performance"],rug=True, bins=20, norm_hist=True,kde_kws={"label":"fixed","clip":(0.0,100.0)})
    q = sns.distplot(df.loc[df["Fixation"] == "no fix","% Performance"],rug=True, bins=20,kde=True, norm_hist=False,kde_kws={"label":"unfixed","clip":(0.0,100.0)})\
        .set(ylabel="Probability density", xlabel="Daily success rate [%]",title="Unpooled daily success rate \n"
                                                                                 "distribution of all mice")
    sns.despine()
    plt.tight_layout()
    plt.show()
def draw_jointplot(df):
    sns.set_context("paper")
    df_groupby_fixation = df.groupby(["Day", "Fixation"])["% Performance"].mean()[
                          :122].unstack().reset_index()
    h = (sns.jointplot(y=df_groupby_fixation["fix"], x=df_groupby_fixation["no fix"],
                       marginal_kws=dict(bins=20, rug=True), data=df_groupby_fixation, kind="reg",
                       ylim={40, 90}, xlim={0, 80})).plot_joint(sns.kdeplot, n_levels=5)
    h.set_axis_labels("Daily success rate [%] \n unfixed trials", "Daily success rate [%] \n fixed trials", fontsize=16)
    h.fig.suptitle("Associated day-aligned and mouse-pooled success rates", fontsize=16)

    # ////////////////////////////////    regression plot
    X = df_groupby_fixation.iloc[:, 2].values.reshape(-1, 1)
    y = df_groupby_fixation.iloc[:, 1].values

    regression = linear_fit(X, y)
    slope = regression.params[0]
    intercept = regression.params[1]
    score = regression.rsquared
    slope_error = regression.bse[0]
    intercept_error = regression.bse[1]
    X_plot = np.linspace(0, 85, 100)
    Y_plot = intercept + X_plot * slope
    plt.plot(X_plot, Y_plot, color='r')
    #text = "linear fit [R$^2$={}]: \n SR(d) = {}($\pm${}) + d $\cdot$ {}($\pm${})  " \
    #   .format(score.round(2), intercept.round(1), intercept_error.round(1), slope.round(2), slope_error.round(2))
    # ///////////////////////////////////////////////////////////

    CORRELATION_COEFFICIENT = df_groupby_fixation["fix"].corr(df_groupby_fixation["no fix"])
    text = "correlation coefficient: " + str(CORRELATION_COEFFICIENT.round(2))
    plt.text(40, 85, text, horizontalalignment='center', size='medium', color='black', weight='semibold')

    plt.tight_layout()
    plt.show()


def st_check(timeseries,df):
    rolmean = df.rolling(window=6).mean()  ## as month is year divide by 12
    rolstd = df.rolling(window=3).std()

    # Plot rolling statistics:
    #orig = plt.plot(timeseries, color='blue', label='Original')
    mean = rolmean.plot(figsize=(8,4))
    #std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()

    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    print(dfoutput)


def make_dataframe(data,detailed):
    if detailed == "No" or detailed == "Yes":
        df = pd.DataFrame(data=data,
                     columns=["Date", "Mouse", "Fixation", "GO", "Outcome", "Trials", "Trials this day","% Performance", "Cage"])
    elif detailed == "pooled":
        df = pd.DataFrame(data=data,
                          columns=["Mouse", "Fixation", "GO", "Outcome", "Trials", "Trials this day","% Performance","AVG % Performance", "STDEV % Performance", "Cage"])

    return df
#//////////////////////////////////////////////////////////////////
table = "together"
graph = "split"
filteredtaglist=["201608466","201608468","201608481","201609136","201609336","210608298","2016080026",
                 "2016090793","2016090943",
                 "2016090629","2016090797","2016090882","2016090964","2016090965","2016090985","2016091183",
                 "201608252","201608423","201608474",
                 "801010270","801010219","801010205"]

taglist=[201608466,201608468,201608481,201609114,201609124,201609136,201609336,210608298,210608315,2016080026,
         2016090636,2016090791,2016090793,2016090845,2016090943,2016090948,2016091033,2016091112,2016091172,2016091184,
         2016090629,2016090647,2016090797,2016090882,2016090964,2016090965,2016090985,2016091183,2016090707,
         201608252,201608423,201608474,2016080008,2016080009,2016080104,2016080242,2016080250,
         801010270,801010219,801010044,801010576,801010442,801010205,801010545,801010462]
cage4taglist= ["201608252","201608423"]
outcomelist = ["GO","NO GO"]
filteredoutcomelist= ["GO"]


#//////////////////////////////////// MAIN FUNCTION ////////////////////////////////////////////////////////////////////////////////////////////
detailed = "No"
query = generateQuery(detailed)                                               #define query in query function above
data = list(getFromDatabase(query))                                           #no need to change
df = make_dataframe(data,detailed)                                            #define dataframe structure in function above. especially column names

df_single_mouse, df_filteredtaglist_go_nogo = prepare_dataframes(df,detailed,filteredoutcomelist)
df_filteredtaglist_go_nogo.to_html("heisenberg.html", col_space=80)
pivoted_table = df_filteredtaglist_go_nogo.pivot_table(values=["% Performance"], columns=['Fixation',"Cage", "Mouse"],index=['Day'],aggfunc=np.sum, fill_value=("0"))
pivoted_table.to_html("anova.html", bold_rows=False, col_space=80)
print(rp.summary_cont(df_filteredtaglist_go_nogo.groupby(["Fixation"]))["% Performance"])



df_groupby_fixation = df_filteredtaglist_go_nogo.groupby(["Day", "Fixation"])["% Performance"].mean().unstack().reset_index()
X =df_groupby_fixation["Day"].values.reshape(-1,1)
y = df_groupby_fixation.iloc[:, 1].values

print(df_groupby_fixation.corr())

dPrime_dataframe()
TWOway_anovaRM(df_filteredtaglist_go_nogo)
make_augmented_dickey_fuller_test(y,"c")
regression = linear_fit(X,y)
save_table(table,df_filteredtaglist_go_nogo)

#print(st_check(y,df_groupby_fixation.iloc[:, 1]))

#draw_doubleplot(df_single_mouse,df_filteredtaglist_go_nogo,"Fixation")
draw_singleplot(df_filteredtaglist_go_nogo,"Cage",None)
#draw_countplot(df_filteredtaglist_go_nogo)
draw_histogram(df_filteredtaglist_go_nogo)
draw_jointplot(df_filteredtaglist_go_nogo)
#plt.gcf().autofmt_xdate()


