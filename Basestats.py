import pandas as pd
import pymysql


filteredtaglist=["201608466","201608468","201608481","201609136","201609336","210608298","2016080026",
                 "2016090793","2016090943",
                 "2016090629","2016090797","2016090882","2016090964","2016090965","2016090985","2016091183",
                 "201608252","201608423","201608474",
                 "801010270","801010219","801010205"]
taglist=[201608466,201608468,201608481,201609114,201609124,201609136,201609336,210608298,210608315,2016080026,
         2016090636,2016090791,2016090793,2016090845,2016090943,2016090948,2016091033,2016091112,2016091172,2016091184,
         2016090629,2016090647,2016090797,2016090882,2016090964,2016090965,2016090985,2016091183,2016090707,
         201608252,201608423,201608474,2016080008,2016080009,2016080104,2016080242,2016080250,
         801010270,801010219,801010044,801010576,801010442,801010205,801010545,801010462,
         801010272, 801010278, 801010378, 801010459, 801010534]


"""
SUM(CASE WHEN `Trial or Entry`= "fix" THEN `counts` ELSE 0 END) AS `Headfixes`
"""




def generateQuery(table):
    if table == "headfixes":
        query = """SELECT a.`Mouse`, a.`cage`,COUNT(DISTINCT `Day`) AS `Days in Cage`, ROUND(COUNT(`Minutes headfixed`),1) AS `Days with headfixation`,
                ROUND(SUM(`Minutes headfixed`)/60,1) AS `Total hours headfixed`, SUM(`counts`) AS `Total headfixes`,
                ROUND(AVG(`counts`),1) AS `Headfixes/day`,ROUND(STDDEV_SAMP(`counts`),1) AS `STD Headfixes/day`,
                ROUND(AVG(`Minutes headfixed`),1) AS `Minutes headfixed/day`, ROUND(STDDEV_SAMP(`Minutes headfixed`),1) AS `STD Minutes headfixed/day`,
                `Reason_for_retirement` FROM
                (SELECT `mice_autoheadfix`.`cage`, `mice_autoheadfix`.`Mouse`, Date(`entries`.`Timestamp`) AS `Day`, (SUM(`Headfix duration`)/60) AS `Minutes headfixed`,
                SUM(IF(`Trial or Entry` = "fix" ,1,NULL)) AS `counts`, `mice_autoheadfix`.`Reason_for_retirement`
                FROM `entries`
                LEFT JOIN `mice_autoheadfix` ON `entries`.`Mouse` = `mice_autoheadfix`.`Mouse`
                WHERE
                ((Date(`entries`.`Timestamp`) BETWEEN "2017-07-12" AND "2017-10-12")
                OR (Date(`entries`.`Timestamp`) BETWEEN "2018-02-14" and "2018-04-01")
                OR (Date(`entries`.`Timestamp`) BETWEEN "2018-04-23" AND "2018-06-01")
                OR (Date(`entries`.`Timestamp`) BETWEEN "2018-07-24" and "2018-10-24")
                OR (Date(`entries`.`Timestamp`) BETWEEN "2018-11-15" AND "2018-12-20"))
                AND `mice_autoheadfix`.`Activity` = "good"
                GROUP BY `Day`,`entries`.`Mouse`)a
                GROUP BY a.`Mouse`
                ORDER BY a.`cage`,a.`Mouse`"""
    elif table == "entries":
        query = """SELECT `cage`,`Mouse`,COUNT(`Day`) AS `Days in Cage`,
                ROUND(AVG(`Minutes in chamber`),1) AS `Minutes in chamber/Day`,ROUND(STDDEV_SAMP(`Minutes in chamber`),1) AS `STD Minutes in chamber/Day`,
                ROUND(AVG(`entries`),1) AS `entries/Day`,ROUND(STDDEV_SAMP(`entries`),1) AS `STD entries/Day`  FROM (SELECT `mice_autoheadfix`.`cage`, `mice_autoheadfix`.`Mouse`, Date(`entries`.`Timestamp`) AS `Day`, (SUM(`Duration`)/60) AS `Minutes in chamber`,
                count(*) AS `entries`
                FROM `entries`
                LEFT JOIN `mice_autoheadfix` ON `entries`.`Mouse` = `mice_autoheadfix`.`Mouse`
                WHERE
                ((Date(`entries`.`Timestamp`) BETWEEN "2017-07-12" AND "2017-10-12")
                OR (Date(`entries`.`Timestamp`) BETWEEN "2018-02-14" and "2018-04-01")
                OR (Date(`entries`.`Timestamp`) BETWEEN "2018-04-23" AND "2018-06-01")
                OR (Date(`entries`.`Timestamp`) BETWEEN "2018-07-24" and "2018-10-24")
                OR (Date(`entries`.`Timestamp`) BETWEEN "2018-11-15" AND "2018-12-20"))
                AND `mice_autoheadfix`.`Activity` = "good"
                GROUP BY `Day`,`Mouse`)a
                GROUP BY a.`Mouse`
                ORDER BY `cage`,`Mouse`"""
    return query

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

def clean_table(df):

    for idx, row in df.iterrows():
        if df.loc[idx, 'Event'] == "SeshStart" or df.loc[idx, 'Event'] == "SeshEnd":
            df.loc[idx, 'Tag'] = "NULL"
        if df.loc[idx, 'Tag'] == 0:
            df.loc[idx, 'Tag'] = df.loc[idx - 1, 'Tag']
        if df.loc[idx, 'Date'] == "reward":
            df.loc[idx,['Tag', 'Unix', 'Event', 'Date']] = df.loc[idx,['Tag', 'Event','Date', 'Unix']].values
    df = df[df.Tag != "NULL"]  # kick all rows with Tag: NULL
    df = df.sort_values(by=['Unix'])
    df = df.drop(labels="Date", axis=1)
    return df

stats = "headfixes"


data = list(getFromDatabase(query = generateQuery(stats)))
if stats == "headfixes":
    df = pd.DataFrame(data=data,
                      columns=["Mouse","Group","Days in cage", "Days with headfixation", "Total hours headfixed", "Total headfixes",
                               "Headfixes/day","STD Headfixes/day","Minutes headfixed / day","STD Minutes headfixed/day",
                               "Reason for retirement"])
elif stats == "entries":
    df = pd.DataFrame(data=data,
                      columns=["Group","Mouse", "Days in cage", "Minutes in chamber/Day", "Entries/Day","STD Entries/Day"])
print(df)
df.to_csv("summarystatsloose{}.csv".format(stats))
