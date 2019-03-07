import sys
import atexit
import platform
import time
import pandas as pd
import numpy as np
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
import pymysql
import seaborn as sns
import six
import glob


from pybloqs import Block
import pybloqs.block.table_formatters as tf
from IPython.core.display import display, HTML
filteredtaglist=["201608466","201608468","201608481","201609136","201609336","210608298","2016080026",
                 "2016090793","2016090943",
                 "2016090629","2016090797","2016090882","2016090964","2016090965","2016090985","2016091183",
                 "201608252","201608423","201608474"]
taglist=[201608466,201608468,201608481,201609114,201609124,201609136,201609336,210608298,210608315,2016080026,
         2016090636,2016090791,2016090793,2016090845,2016090943,2016090948,2016091033,2016091112,2016091172,2016091184,
         2016090629,2016090647,2016090797,2016090882,2016090964,2016090965,2016090985,2016091183,2016090707,
         201608252,201608423,201608474,2016080008,2016080009,2016080104,2016080242,2016080250]
#table = "sessiontrials"
table = "sessiontime"
def getFromDatabase(query):
    db2 = pymysql.connect(host="localhost", user="root", db="murphylab", password='autohead2015')
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
    if table == "sessiontrials":
        query = """SELECT p.`Mouse`,p.`Fixation`,p.`sumtrials`,a.`correctcounts`, round((a.`correctcounts`/p.`sumtrials`)*100,2) AS '% Performance' FROM
                        (SELECT s.`Mouse`,s.`Trial in session`,s.`Fixation`,COUNT(*) AS `sumtrials` FROM(SELECT k.`Dates` FROM(SELECT COUNT(t.`Fixation`),t.`Fixation`, t.`Dates` FROM (select Date(`Trial start`) as `Dates`, `Fixation` from `headfix_trials_summary`
                         group by Date(`Trial start`), `Fixation`)t
                         group by t.`Dates`)k
                         where k.`Fixation`='fix')x
                         INNER JOIN
                        (SELECT `Mouse`,`Fixation`,`Trial in session`,Date(`Trial start`) AS `Datess` FROM `headfix_trials_summary` WHERE `Task` = 'GO in time window' AND (Date(`Trial start`) < '2018-07-01' OR Date(`Trial start`) > '2018-08-05')) s
                        ON x.`Dates` = s.`Datess`
                        GROUP BY `Mouse`,`Trial in session`,`Fixation`)p
                        INNER JOIN
                        (SELECT s.`Mouse`,s.`Trial in session`,s.`Fixation`,COUNT(*) AS `correctcounts` FROM(SELECT k.`Dates` FROM(SELECT COUNT(t.`Fixation`),t.`Fixation`, t.`Dates` FROM (select Date(`Trial start`) as `Dates`, `Fixation` from `headfix_trials_summary`
                         group by Date(`Trial start`), `Fixation`)t
                         group by t.`Dates`)k
                         where k.`Fixation`='fix')x
                         INNER JOIN
                        (SELECT `Mouse`,`Fixation`,`Trial in session`,Date(`Trial start`) AS `Datess` FROM `headfix_trials_summary` WHERE `Task` = 'GO in time window' AND `Outcome`='correct'AND (Date(`Trial start`) < '2018-07-01' OR Date(`Trial start`) > '2018-08-05')) s
                        ON x.`Dates` = s.`Datess`
                        GROUP BY `Mouse`,`Trial in session`,`Fixation`)a
                        ON  p.`Mouse`=  a.`Mouse` AND p.`Trial in session`=a.`Trial in session` AND p.`Fixation`=a.`Fixation`"""
        return query
    if table == "sessiontime":
        query = """select
                    `Mouse`,`Headfix duration at stimulus`,`Task`,`Fixation`,`Notes`
                    from `headfix_trials_summary` where
                    ((Date(`Trial start`) between "2017-08-28" and "2017-10-12") OR (Date(`Trial start`) between "2018-02-19" and "2018-04-01")
                     OR (Date(`Trial start`) between "2018-04-23" and "2018-06-01") OR (Date(`Trial start`) between "2018-08-08" and "2018-10-06"))
                     AND `Task` = "GO in time window" """
        return query

query = generateQuery(table)
data = list(getFromDatabase(query))
if table == "sessiontrials":
    df = pd.DataFrame(data=data,columns = ["Mouse","Trial in session","Fixation","Trials","correct Trials","% Performance"])
    df = df[df["Mouse"].isin(filteredtaglist)]
    print(df)
    g = sns.catplot(x="Trial in session", y="% Performance", hue="Fixation", col= "Mouse", col_wrap = 5, data=df, kind="bar", height=2.5, aspect=.8)

if table == "sessiontime":
    df = pd.DataFrame(data=data,columns = ["Mouse","Headfix duration at stimulus","Task","Fixation","Notes"])
    df = df[df["Mouse"].isin(filteredtaglist)]
    df.replace({"Notes": {"GO=2": "correct", "GO=-2": "fail", "GO=-4": "fail",
                            "GO=1": "correct no go", "GO=-1": "fail: licked, but right time window",
                            "GO=-3": "fail:licked and wrong time window"}}, inplace=True)
    #df["Headfix duration at stimulus"] = df["% Performance"].convert_objects(convert_numeric=True)
    print(df)
    g = sns.violinplot(x="Fixation", y="Headfix duration at stimulus", hue="Notes",data=df, split=True,inner="quarts")

sns.set(style="whitegrid")
sns.despine(left=True)
# g.legend(['Fix', 'No fix'], bbox_to_anchor=(0.5, 1.15), borderaxespad=0., title=mouse, frameon=False, ncol=2)
#g.set_ylabels("% Performance")
plt.show()