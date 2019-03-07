import sys
import atexit
import platform
import time
import pandas as pd
import numpy as np
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
import pymysql
from password import database_password as DBpwd
import glob

filteredtaglist = ["201608466", "201608468", "201608481", "201609136", "201609336", "210608298", "2016080026",
                   "2016090793", "2016090943",
                   "2016090629", "2016090797", "2016090882", "2016090964", "2016090965", "2016090985", "2016091183",
                   "201608252", "201608423", "201608474",
                   "801010270", "801010219", "801010044", "801010576", "801010442", "801010205", "801010545","801010462",
                   "801010272", "801010278", "801010378", "801010459", "801010534","801010543","801010546"]



def generate_commands(condition):
    if condition == "no":
        query = """SELECT Date(`Timestamp`),`Mouse`, COUNT(*) FROM `rewards` WHERE `Timestamp` >= (Date(NOW()) - 2) 
                    Group BY Date(`Timestamp`),`Mouse`"""
    elif condition == "yes":
        query = """SELECT Date(`Timestamp`),`Mouse`,`Reward_type`, COUNT(*) FROM `rewards` WHERE `Timestamp` >= (Date(NOW()) - INTERVAL 8 HOUR)  
                            Group BY Date(`Timestamp`),`Mouse`,`Reward_type`"""
    return query

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



data = list(getFromDatabase(query = generate_commands("yes")))
df = pd.DataFrame(data=data,columns=["Date","Mouse","condition","Waterdrops"])
print(df)
df.to_csv("water+.csv")
data = list(getFromDatabase(query = generate_commands("no")))
df = pd.DataFrame(data=data,columns=["Date","Mouse","Waterdrops"])
print(df)
df.to_csv("water-.csv")