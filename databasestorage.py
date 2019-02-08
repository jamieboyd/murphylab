"""#!/usr/bin/env python"""
import pandas as pd
import pymysql
import glob

project = 1
notes = ""

def generate_commands():
    query="""INSERT INTO `textfilesgroup5`
    (`Tag`, `Timestamp`, `Event`, `Project_ID`, `Notes`)
        VALUES(%s,FROM_UNIXTIME(%s),%s,%s,%s)"""
    values= (tag,unix,event,project,notes)
    return query, values

def saveToDatabase(query, values):
    db1 = pymysql.connect(host="localhost",user="root",db="murphylab",password='password')
    cur1 = db1.cursor()
    try:
        cur1.execute(query, values)
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
        return Nonecd
    except ValueError as e:
        print("MySQL Error: ValueError: %s" % str(e))
        return None
    db1.close()

#MAIN FUNCTION

#choose files
allfiles = glob.glob('D:/Cagedata/textfiles/Group5_textFiles/todo/*.txt')

for f in allfiles:
    df = pd.read_csv(f, sep="\t",header=None, names = ["Tag", "Unix", "Event", "Date"])
    df = df.sort_values(by=['Unix'])
    print(f)
    project = f[27:28]
    df = df.drop(labels="Date", axis=1)
    #replace zero tags for the licks
    for idx, row in df.iterrows():
        if  df.loc[idx,'Event'] == "SeshStart" or df.loc[idx,'Event'] == "SeshEnd":
            df.loc[idx,'Tag'] = "NULL"
        if df.loc[idx, 'Tag'] == 0:
             df.loc[idx, 'Tag'] = df.loc[idx - 1, 'Tag']

    array = df.values.tolist()
    print(len(array))
    for i in range(len(array)):
        tag = array[i][0]
        unix = array[i][1]
        event = array[i][2]
        query, values = generate_commands()
        saveToDatabase(query, values)

print("done")
