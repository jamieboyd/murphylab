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
    if table == "detailed":
        query = """SELECT t.`ts1`,t.`Mouse`, t.`Fixation` ,t.`Task`,t.`Outcome`,t.`Notes`,
        t.`Trials`,s.`Sumtrials`as 'Trials this day' ,ROUND(t.`Trials`/Sum(s.`Sumtrials`)*100,2) as '% Performance', t.`Project_ID` FROM
        (select
        Date(`Trial start`) as `ts1`,`Mouse`,count(*) as `Trials`,`Task`,`Outcome`,`Fixation`,`Notes`
        from `headfix_trials_summary` WHERE
        `Trial start` between "2018-08-21 00:00:00.01" and "2018-09-03 23:59:59.99"
        group by `ts1`, `Mouse`,`Outcome`, `Fixation`) t
        INNER JOIN (SELECT `ts1`,`Mouse`, `Fixation`,`Task`,Sum(`Trials`) as `Sumtrials` FROM
        (SELECT Date(`Trial start`) as `ts1`,`Mouse`,count(*) as `Trials`,`Task`,`Outcome`,`Fixation`,`Notes`
        from `headfix_trials_summary` WHERE
        `Trial start` between "2018-08-21 00:00:00.01" and "2018-09-03 23:59:59.99"
        group by `ts1`, `Mouse`,`Outcome`, `Fixation`) k
        GROUP by `ts1`,`Mouse`, `Fixation`,`Task`
        ORDER by `ts1`,`Mouse`,`Task`) s ON s.`Mouse` = t.`Mouse` AND s.`ts1` = t.`ts1` AND s.`Fixation` = t.`Fixation` AND s.`Task` = t.`Task`
        GROUP by `ts1`,`Mouse`, `Fixation`,`Outcome`
        ORDER by `ts1`,`Mouse`, `Fixation`,`Task`,`Outcome` """
    elif table =="not detailed":
        query = """SELECT t.`ts1`,t.`Mouse`,t. `Fixation`, t.`Task`,t.`Notes`,
        t.`Trials`,s.`Sumtrials`as 'Trials this day' ,ROUND(t.`Trials`/Sum(s.`Sumtrials`)*100,2) as '% Performance',t.`Project_ID` FROM
        (select
        Date(`Trial start`) as `ts1`,
        `Mouse`,count(*) as `Trials`,`Task`,`Fixation`,`Notes`,`Project_ID`
        from `headfix_trials_summary` where
        `Trial start` between "2017-08-23 00:00:00.01" and "2018-09-28 23:59:59.99"
        group by `ts1`, `Mouse`,`Notes`, `Fixation`) t
        INNER JOIN (SELECT `ts1`,`Mouse`, `Fixation`,`Task`,Sum(`Trials`) as `Sumtrials` FROM
        (select
        Date(`Trial start`) as `ts1`,`Mouse`,count(*) as `Trials`,`Task`,`Fixation`,`Notes`
        from `headfix_trials_summary` where
        `Trial start` between "2017-08-23 00:00:00.01" and "2018-09-28 23:59:59.99"
        group by `ts1`, `Mouse`,`Notes`, `Fixation`) k
        GROUP by `ts1`,`Mouse`, `Fixation`,`Task`
        ORDER by `ts1`,`Mouse`,`Task`) s ON s.`Mouse` = t.`Mouse` AND s.`ts1` = t.`ts1` AND s.`Fixation` = t.`Fixation` AND s.`Task` = t.`Task`
        GROUP by `ts1`,`Mouse`, `Fixation`,`Notes`
        ORDER by `ts1`,`Mouse`, `Fixation`,`Task`,`Notes`"""
    elif table == "split":
        query = """SELECT t.`ts1`,t.`Mouse`,t. `Fixation`, t.`Task`,t.`Notes`,
                t.`Trials`,s.`Sumtrials`as 'Trials this day' ,ROUND(t.`Trials`/Sum(s.`Sumtrials`)*100,2) as '% Performance',t.`Project_ID` FROM
                (select
                Date(`Trial start`) as `ts1`,
                `Mouse`,count(*) as `Trials`,`Task`,`Fixation`,`Notes`,`Project_ID`
                from `headfix_trials_summary` where
                `Trial start` between "2017-08-23 00:00:00.01" and "2018-12-20 23:59:59.99"
                group by `ts1`, `Mouse`,`Notes`, `Fixation`) t
                INNER JOIN (SELECT `ts1`,`Mouse`, `Fixation`,`Task`,Sum(`Trials`) as `Sumtrials` FROM
                (select
                Date(`Trial start`) as `ts1`,`Mouse`,count(*) as `Trials`,`Task`,`Fixation`,`Notes`
                from `headfix_trials_summary` where
                `Trial start` between "2017-08-23 00:00:00.01" and "2018-12-20 23:59:59.99"
                group by `ts1`, `Mouse`,`Notes`, `Fixation`) k
                GROUP by `ts1`,`Mouse`, `Fixation`,`Task`
                ORDER by `ts1`,`Mouse`,`Task`) s ON s.`Mouse` = t.`Mouse` AND s.`ts1` = t.`ts1` AND s.`Fixation` = t.`Fixation` AND s.`Task` = t.`Task`
                GROUP by `ts1`,`Mouse`, `Fixation`,`Notes`
                ORDER by `ts1`,`Mouse`, `Fixation`,`Task`,`Notes`"""
    return query



#MAIN FUNCTION

table = "split"
#table = "sessiontrials"
query = generateQuery(table)
data = list(getFromDatabase(query))
if table == "detailed":
    df = pd.DataFrame(data=data,columns = ["Day","Mouse","Fixation","Task","Outcome","Notes","Trials","Trials this day","% Performance"])
    df1 = df.drop(["Notes","Trials","Trials this day"],axis=1)
    pivoted_table = df.pivot_table(values=["% Performance"], columns=['Fixation', 'Task', "Outcome"],
                                   index=['Mouse', 'Day'],
                                   aggfunc=np.sum, fill_value=("0"))
    pivoted_table.to_html("detailed.html", bold_rows=False, col_space=80)
elif table == "not detailed":
    df = pd.DataFrame(data=data,
                      columns=["Day", "Mouse", "Fixation", "Task", "Outcome", "Trials", "Trials this day",
                               "% Performance","Cage"])
    df.replace({"Outcome":{"GO=2": "correct","GO=-2": "fail: no licks","GO=-4": "fail: licked before stimulus ended",
                           "GO=1": "correct","GO=-1": "fail: licked, but right time window", "GO=-3": "fail:licked and wrong time window"}},inplace=True)
    df1 = df.drop([ "Trials this day"], axis=1)
    df1["% Performance (n Trials)"] = df1['% Performance'].map(str) + "  (" + df1['Trials'].map(str) + ")"
    pivoted_table = df1.pivot_table(values=['% Performance (n Trials)'], columns=['Fixation', 'Task', "Outcome"],
                                   index=['Cage','Mouse', 'Day'],
                                   aggfunc=lambda x: ' '.join(x), fill_value=("0"))
elif table == "split":
    df = pd.DataFrame(data=data,
                      columns=["Day", "Mouse", "Fixation", "Task", "Outcome", "Trials", "Trials this day",
                               "% Performance","Cage"])
    df.replace({"Outcome":{"GO=2": "correct","GO=-2": "fail: no licks","GO=-4": "fail: licked before stimulus ended",
                           "GO=1": "correct","GO=-1": "fail: licked, but right time window", "GO=-3": "fail:licked and wrong time window"}},inplace=True)
    df1 = df.drop([ "Trials this day"], axis=1)
    pivoted_table = df1.pivot_table(values=['% Performance','Trials'], columns=['Fixation', 'Task', 'Outcome'],
                                   index=['Cage','Mouse', 'Day'],aggfunc={'% Performance':np.sum,'Trials':np.sum},
                                 fill_value=0, margins=True)
    column_order = ['Fixation', 'Task', 'Outcome']
    pivoted_table.columns = pivoted_table.columns.swaplevel(3,0).swaplevel(2,0).swaplevel(1,0)
    pivoted_table.sort_index(1,inplace=True)
    pivoted_table.drop(('All',"","",'% Performance'), axis = 1, inplace = True)
pivoted_table.to_html("split.html",  col_space=80)

for mouse in filteredtaglist:
    print(mouse)
    df2 = df1[df1["Mouse"] == str(mouse)]
    print(df2)
    """
    testdf = df2.pivot_table(values = ["% Performance"], columns=["Fixation","Task","Outcome"],index=['Day'],
                                       aggfunc=np.sum, fill_value=("0"))
    testdf.columns = ["|".join((i,j,k,l))for i,j,k,l in testdf.columns]
    testdf.reset_index()
    testdf.to_html("testdf.html")
    
    cols=[]
    for i in range(len(testdf.columns)):
        if "correct" not in testdf.columns[i]:
            cols.append(i)
    testdf.drop(testdf.columns[cols] ,axis =1,inplace=True )

    testdf = pd.DataFrame(testdf.to_records())
    #sns.lineplot(data=testdf, palette="tab10", linewidth=2.5)


    sns.set(style="ticks")

    col = list(testdf.iloc[:,1:].columns.values)
    try:
        dates = pd.DatetimeIndex(testdf['Day'].values)
        values = np.array(testdf.iloc[:,1:])
        data = pd.DataFrame(values, dates, columns=col,dtype=float)
        #print(data)
        bild = sns.lineplot(data=data, palette="Paired", linewidth=2.5, dashes=False)
        bild.set(ylabel = "Successrate [%] of GO-trials")
        bild.legend(['Fix','No fix'],bbox_to_anchor=(0.5,1.15),borderaxespad=0.,title=mouse,frameon=False,ncol=2 )
        sns.despine()
        #bild.set_title(mouse)

        plt.gcf().autofmt_xdate()
        plt.show()
    except:
        continue
"""
print(df)

sns.despine()
plt.gcf().autofmt_xdate()
plt.show()