"""#!/usr/bin/env python"""
import sys
import atexit
import platform
import time
import pandas as pd
import numpy as np
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
import pymysql
import glob
from password import database_password as DBpwd

NOTES = ""
PROJECT = 1

SESSION_START = 0
FIXATION ="fix"
TRIAL_START = 0
TRIAL_IN_SESSION =0
TASK =""
STIMULUS = ""
OUTCOME = ""
LICKS_AFTER_STIM =0
TRIAL_DURATION = 0
HEADFIX_DURATION = 0
VIDEO = ""
LICKWITHHOLD_TIME=0
LED_ON = None
LED_OFF = None
VIDEO_NOTES = ""
LICK_DURATION = 0                                             #Lick duration after stimulus
REACTION_TIME = 0                                             #time till first lick after stimulus
LICKS_TO_TRIGGER_REWARD = 0                                   #number of licks between stimulus and reward given
REWARD_DELAY = 0                                              #time between a trial start and release of water in a trial
LICKS_AFTER_REWARD = 0                                        # number of licks to drink water
LICK_DURATION_AFTER_REWARD = 0                                #time spent licking after reward
REWARD = "NO"                                                 # indicator if reward was earned
LICKS_WHILE_HEADFIXED = 0
LICKS_WITHIN_LICKWITHHOLD_TIME = None

trial_in_session_counter=0                                    #counts up how many trials are done in a session, increases with every trial
lick_counter_trials =0                                        # counts the number of licks in a trial, counts up during a the time frame of a trial
last_lick_time=0                                              #timestamp of the last occuring lick
lick_time_start=0
reaction_time_started = False
reaction_time_start = 0
result = 0
reward = False
real_trial = False
previous_outcome = -2
previous_outcome_licked = False
lick_counter_headfix = 0

TAG = "zero"
ENTRY_TIME = 0
ENTRY_DURATION = 0
ENTRY_TYPE = "entry"
HEADFIX_DURATION_ENTRIES = 0
LICKS = 0
LAST_MOUSE = "zero"
LAST_MOUSE_TIME = 0
LAST_MOUSE_HEADFIX = None
LAST_MOUSE_TIME_HEADFIX = 0
lick_counter_entries =0
started = False
last_mouse_time_start = 0
last_mouse_time_headfix_start = 0
headfix_start = 0

water_available = False
ENTER_ID = 0
TRIAL_ID =0

taglist=[201608466,201608468,201608481,201609114,201609124,201609136,201609336,210608298,210608315,2016080026,
         2016090636,2016090791,2016090793,2016090845,2016090943,2016090948,2016091033,2016091112,2016091172,2016091184,
         2016090629,2016090647,2016090797,2016090882,2016090964,2016090965,2016090985,2016091183,2016090707,
         201608252,201608423,201608474,2016080008,2016080009,2016080104,2016080242,2016080250,
         801010270,801010219,801010044,801010576,801010442,801010205,801010545,801010462,
         801010272,801010278,801010378,801010459,801010534,801010543,801010546]

stimulus_list=["Buzz:N=1,length=0.50,period=0.60","Buzz:N=3,length=0.10,period=0.20"]

def generate_commands(table):
    if table == "trials_headfixation":
        query="""INSERT INTO `headfix_trials_summary`
        (`Project_ID`, `Mouse`, `Fixation`, `Trial start`, `Trial in session`,`Lickwithhold time`, `Task`, `Stimulus`,
        `Outcome`,`Reaction time`,`Licks after stimulus`, `Lick duration after stimulus`,`Trial duration`,
        `Headfix duration at stimulus`, `Videofile`,`Notes`,`Licks to trigger reward`,`Delay till reward`,`Licks after reward`,
         `Lick duration after reward`,`Reward earned`,`Licks_within_lickwithhold_time`)
            VALUES(%s,%s,%s,FROM_UNIXTIME(%s),%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
        values= (PROJECT,TAG,FIXATION,TRIAL_START,TRIAL_IN_SESSION,LICKWITHHOLD_TIME,TASK, STIMULUS,OUTCOME,REACTION_TIME,
                 LICKS_AFTER_STIM,LICK_DURATION,TRIAL_DURATION,HEADFIX_DURATION,VIDEO, NOTES, LICKS_TO_TRIGGER_REWARD,
                 REWARD_DELAY, LICKS_AFTER_REWARD, LICK_DURATION_AFTER_REWARD, REWARD,LICKS_WITHIN_LICKWITHHOLD_TIME)

    elif table == "videos":
        query = """INSERT INTO `videos_list`
        (`Videofile`,`Session start`,`LED_on`,`LED_off`,`Fixation`,`Mouse`,`Notes`)
            VALUES(%s,FROM_UNIXTIME(%s),FROM_UNIXTIME(%s),FROM_UNIXTIME(%s),%s,%s,%s)"""
        values= (VIDEO,SESSION_START,LED_ON,LED_OFF,FIXATION,TAG,VIDEO_NOTES)

    elif table == "entries":
        query = """INSERT INTO `entries`
                (`Project_ID`, `Mouse`, `Timestamp`, `Duration`,`Trial or Entry`, `Headfix duration`, `Licks`,`Licks_while_headfixed`,
                `Last Mouse`,`Time after last Mouse exits`,`Last mouse headfixed`,`Time since last headfix`)
                    VALUES(%s,%s,FROM_UNIXTIME(%s),%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
        values = (PROJECT, TAG, ENTRY_TIME, ENTRY_DURATION, ENTRY_TYPE, HEADFIX_DURATION_ENTRIES, LICKS,LICKS_WHILE_HEADFIXED, LAST_MOUSE, LAST_MOUSE_TIME,LAST_MOUSE_HEADFIX,LAST_MOUSE_TIME_HEADFIX)
    elif table == "rewards":
        query = """INSERT INTO `Rewards`
                (`Mouse`,`Timestamp`,`Reward_type`,`Related_trial`,`Related_entry`)
                VALUES(%s,FROM_UNIXTIME(%s),%s,FROM_UNIXTIME(%s),FROM_UNIXTIME(%s))"""
        values = (TAG,REWARD_TIME, ENTRY_TYPE,TRIAL_START,ENTRY_TIME)
    elif table == "licks":
        query = """INSERT INTO `licks`
                (`Mouse`,`Timestamp`,`Related_trial`,`Delta_time`)
                VALUES(%s,FROM_UNIXTIME(%s),FROM_UNIXTIME(%s),%s)"""
        values = (TAG,LICKTIME,TRIAL_START, DELTA_TIME)
    else:
        print("Error in table selection")
    return query, values

def saveToDatabase(table):
    query, values = generate_commands(table)
    db1 = pymysql.connect(host="localhost",user="root",db="murphylab",password=DBpwd)
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
        return None
    except ValueError as e:
        print("MySQL Error: ValueError: %s" % str(e))
        return None
    db1.close()

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

def check_result(result,event):
    global TASK
    global OUTCOME
    global STIMULUS
    global previous_outcome_licked
    global previous_outcome

    if "Buzz:N=1" in event:                                       #update stimulus list because of currupted textfiles
        stimulus_list[0] = event[21:53]
    elif "Buzz:N=2" in event:
        stimulus_list[0] = event[21:53]
    elif "Buzz:N=3" in event:
        stimulus_list[1] = event[21:53]

    if result % 2 == 0:                                           #even numbers are related to GO trials
        TASK = "GO in time window"
        STIMULUS = stimulus_list[0]
    else:                                                         #odd numbers are NO GO
        TASK = "NO GO"
        STIMULUS = stimulus_list[1]

    if result == 0:                                               #overwrite with the special case of GO=0
        TASK = "GO and no licking required"
    if result > 0:                                                #correct trials are positive
        OUTCOME = "correct"
    else:                                                         #name the outcomes of the fail trials
        if result == -4:
            OUTCOME = "fail: licked too early"
        elif result == -1:
            OUTCOME = "fail: licked, but right time window"
        elif result == -3:
            OUTCOME = "fail: licked and also too early"
        elif result == 0:
            OUTCOME = "no licking required"

def save_previous_trial(result, next_outcome,unix,lw2):
    global LICKS_AFTER_STIM
    global LICK_DURATION
    global REACTION_TIME
    global OUTCOME
    global TRIAL_DURATION
    global lick_counter_trials
    global NOTES
    global LICKS_AFTER_REWARD
    global LICKS_TO_TRIGGER_REWARD
    global LICK_DURATION_AFTER_REWARD
    global REWARD_DELAY
    global REWARD
    global reward
    global previous_outcome_licked
    global previous_outcome


    LICKS_AFTER_STIM = lick_counter_trials                                                 #finalyze the count of licks after stimulus
    LICK_DURATION = last_lick_time - lick_time_start                                       #time between last lick and first lick after reward, might be overwritten of no licks occur
    # if trial was correct
    if reward == True:                                                                     #only rewarded trials. they contain licks, otherwise they wouldn't be rewarded
        LICKS_AFTER_REWARD = LICKS_AFTER_STIM - LICKS_TO_TRIGGER_REWARD                    #calculate count of licks after rewards
        if LICKS_AFTER_REWARD != 0:                                                        #if there were licks after reward calculate the lick duration
            try:                                                                           #program will fail if one of the variables is None
                LICK_DURATION_AFTER_REWARD = LICK_DURATION + REACTION_TIME - REWARD_DELAY
            except:
                LICK_DURATION_AFTER_REWARD = None                                          #in case of program failure there won't be a lick time to calculate, because of missing licks
        else:
            LICK_DURATION_AFTER_REWARD = None                                              #no licks no lick time
        REWARD = "YES"                                                                     #this marks if the reward was earned, not if it was taken
    # trial not correct
    else:
        LICKS_AFTER_REWARD = None
        LICKS_TO_TRIGGER_REWARD = None
        LICK_DURATION_AFTER_REWARD = None
        REWARD_DELAY = None
        REWARD = "NO"
    reward = False                                                                         #after the previous evaluation reset the parameter

    if reaction_time_started == True:                                                      #mouse did not lick in between, it may be not active
        REACTION_TIME = None
        LICKS_AFTER_STIM = 0
        LICK_DURATION = 0
        if result == -2:                                                                   #two options: correct NO GO trial or incorrect GO trial
            OUTCOME = "fail: did nothing"                                                  #finalyze the outcome for DB
            previous_outcome_licked = False                                                #mark the trial as a trial without licking for activity evaluation
        elif result == 1:                                                                  #correct no go trial mus be further investigated
            if (previous_outcome != -2 and previous_outcome !=1) and (next_outcome != -2 and next_outcome !=1):   #activity prooven
                OUTCOME = "correct: is active"
            elif (previous_outcome == -2 or previous_outcome ==1) and (next_outcome == -2 or next_outcome ==1):   #both surrounding trials had no licks
                OUTCOME = "correct: probably inactive"
            else:                                                                                                 #one of the surrounding trials had no licks
                OUTCOME = "correct: maybe inactive"
    else:                                                                                   #mouse licked
        if result == -2:
            OUTCOME = "fail: licked too late"
            previous_outcome_licked = True
        if result == 1:
            OUTCOME = "correct: is active"                                                  #mouse wasn't supposed to lick, but licked late enough to get a correct NO GO trial
    TRIAL_DURATION = unix - TRIAL_START + LICKWITHHOLD_TIME - lw2                           #lw2 is the lickwithhold time of the current trial
    previous_outcome = result
    saveToDatabase("trials_headfixation")                                                   #save trial information in trials table
    clear_variables("trials",unix)                                                          #reset trial related variables

def clear_variables(variableset,unix):
    #trial related globals
    global LICKS_AFTER_STIM
    global LICK_DURATION
    global REACTION_TIME
    global OUTCOME
    global TRIAL_DURATION
    global lick_counter_trials
    global NOTES
    global LICKS_AFTER_REWARD
    global LICKS_TO_TRIGGER_REWARD
    global LICK_DURATION_AFTER_REWARD
    global REWARD_DELAY
    global REWARD
    global reward
    #entry related globals
    global lick_counter_entries
    global LAST_MOUSE
    global LAST_MOUSE_HEADFIX
    global started
    global HEADFIX_DURATION_ENTRIES
    global last_mouse_time_start
    global last_mouse_time_headfix_start
    global LAST_MOUSE_TIME
    global LAST_MOUSE_TIME_HEADFIX
    global headfix_start
    global water_available
    global lick_counter_headfix
    global LICKS_WHILE_HEADFIXED
    global LICKS_WITHIN_LICKWITHHOLD_TIME

    if variableset == "trials":
        lick_counter_trials = 0
        LICKS_AFTER_STIM = 0
        LICKS_TO_TRIGGER_REWARD = 0
        NOTES = ""
        LICKS_AFTER_REWARD = 0
        LICK_DURATION_AFTER_REWARD = 0
        REWARD_DELAY = 0
        TRIAL_DURATION = 0
        REWARD = "NO"
        reward = False
        LICKS_WITHIN_LICKWITHHOLD_TIME = None
    elif variableset == "entries":
        headfix_start = 0
        lick_counter_entries = 0
        LAST_MOUSE = TAG
        if ENTRY_TYPE == "fix" or ENTRY_TYPE == "nofix":
            LAST_MOUSE_HEADFIX = TAG
            LAST_MOUSE_TIME_HEADFIX = 0
            last_mouse_time_headfix_start = unix
        started = False
        HEADFIX_DURATION_ENTRIES = None
        LAST_MOUSE_TIME = 0
        last_mouse_time_start = unix
        lick_counter_headfix = 0
        LICKS_WHILE_HEADFIXED = 0

def standardize_trial_event_string(event_string):
    #done because of older textfiles which have another syntax
    if event_string[54:57] == "GO=":                                          #up to date syntax
        event = event_string
    if event_string[:4] == "Buzz":                                            #lickwithhold statement is missing mostly problem of early cage 1 and 5
        event = "lickWitholdTime=1.00,"                                       #manually adding this statement and an artificial time!


    if len(event_string) == 53:
        event = event_string + ",GO=0"                                        #this issue occured in cage 5 when licking wasn't required, yet.
    elif len(event) == 21:
        event = event + "Buzz:N=2,length=0.10,period=0.20,GO=0"               #this issue occured at the very beginning of cage 1 when stimulus was implemented
    elif len(event+event_string) == 53:
        event = event + event_string + ",GO=0"                                #this issue occured in cage 1 when licking wasn't required, yet.
    return event


#MAIN FUNCTION variables importatant for save in Database are capital

allfiles = glob.glob('D:/Cagedata/textfiles/Group6_textFiles/todo/*.txt')
#allfiles = glob.glob('D:/Cagedata/textfiles/Group[1-6]_textFiles/*.txt')
#allfiles = glob.glob('D:/Cagedata/textfiles/Group1_textFiles/headFix_2_20170717.txt')
print(allfiles)
for f in allfiles:
    df = pd.read_csv(f, sep="\t",header=None, names = ["Tag", "Unix", "Event", "Date"])
    print(f)
    #PROJECT = int(f[42:43])                                                   #careful, requires counted position in the path
    PROJECT = f[27:28]
    print(PROJECT)
    df = clean_table(df)
    df.to_csv("temp.csv",sep="\t",header=None)
    df1 = pd.read_csv("temp.csv", sep="\t",header=None,names = ["Tag", "Unix", "Event"])
    array = df1.values.tolist()
    df1.to_csv("testi.csv")
    print(len(array))

    for i in range(len(array)):                                               #read in a line of textfile. each line conisists of a mouse-TAG, a unix timestamp and an event
        TAG = int(array[i][0])
        unix = array[i][1]
        event = str(array[i][2])
        #filter mouse and currupted files
        if TAG not in taglist:
            continue                                                          #get rid of nonsense tags
        if unix < 1008915797:
            continue                                                          #get rid of (hopefully all) currupted timestamps
        #start analyzing
        else:
            if "Buzz" in event:
                event = standardize_trial_event_string(event)                 #deal with old textfiles
            if event == "entry":
                had_session = False                                           #indicates that mouse had no session yet. important for detection of double sessions
                ENTRY_TIME = unix                                             #catches time when the mouse enters
                started = True                                                #boolean to prevent irritation for the rest of the program if an entry is missing
                ENTRY_TYPE = "entry"                                          #type will be overwritten when headfixing attempt occurs
                if last_mouse_time_start != 0:                                #condition prevents weird things after restarts
                    LAST_MOUSE_TIME = unix - last_mouse_time_start            #calculate how long it has been since the last mouse left the chamber
                if last_mouse_time_headfix_start != 0:                        #calculate how long it has been since the last mouse left the chamber after a headfix session (including no fix trials)
                    LAST_MOUSE_TIME_HEADFIX = unix - last_mouse_time_headfix_start
            if event == "entryReward":
                water_available = True
            if event == "check No Fix Trial" or event == "check+":            #start of headfixation
                if had_session == True:                                       #condition indicates a double trial without leaving. simulating new entry for documentation
                    ENTRY_DURATION = unix - ENTRY_TIME
                    LICKS = lick_counter_entries
                    saveToDatabase("entries")
                    clear_variables("entries", unix)
                    ENTRY_TIME = unix
                    started = True
                    if last_mouse_time_start != 0:
                        LAST_MOUSE_TIME = unix - last_mouse_time_start
                    if last_mouse_time_headfix_start != 0:
                        LAST_MOUSE_TIME_HEADFIX = unix - last_mouse_time_headfix_start
                trial_in_session_counter = 0                                  #tracks the number of trials in a session
                SESSION_START = unix                                          #time when the mouse get's headfixed
                previous_outcome = -2                                         #initializes this variable . prevents program error during first trial in session. will be overwritten when second trial is in line
                previous_outcome_licked = False                               #initializes this variable
                if event == "check+":
                    FIXATION = "fix"
                    headfix_start = unix                                      #time when headfixing starts
                    ENTRY_TYPE = "fix"                                        #overwrite type from entry to fix
                if event == "check No Fix Trial":
                    FIXATION = "no fix"
                    ENTRY_TYPE = "nofix"                                      #overwrite type from entry to no fix
                    HEADFIX_DURATION_ENTRIES = None                           #this varaiable is the complete headfix duration of the session and is stored in the entries table. differs from HEADFIX_DURATION which is the time how long the mouse is fixed at this moment
            if event == "check-" and ENTRY_TYPE == "entry":
                ENTRY_TYPE = "away"                                           #overwrite type from entry to symbolyze that the mouse fled when it heard the motor
                HEADFIX_DURATION_ENTRIES = None
            if event == "BrainLEDON":
                LED_ON = unix
            if event == "BrainLEDOFF":
                LED_OFF = unix
            if "video" in event:
                VIDEO = "M"+event[6:]                                         #reconstructs the video filename for the videos table
            if event == "reward":                                             #these paramters are only not None when there is a successful GO trial
                LICKS_TO_TRIGGER_REWARD = lick_counter_trials                 #number of licks the mouse made between stimulus and reward
                REWARD_DELAY = unix - TRIAL_START                             #time between reward and stimulus
                reward = True
                water_available = True
            #trial summaries: this get's complicated. biggest problem is that most variables for the trial are calculated after the summary was printed in the textfile
            # this requires a careful management of the flow of the code, by first saving the previous trial and then actualizing the variables for the next trial
            # it requires to partly save some variable values of the trial summary of the textfile in temporary variables
            if "lickWith" in event:
                real_trial = True                                             #marker for the program to pay attention that a trial is in progress
                next_outcome = int(event[57:])                                #store the outcome of trial in a temporary variable
                # save the previous trial
                if trial_in_session_counter !=0:                              #if there was a previous trial then save it now. afterwards textfile entry will be used to save the corrosponding trial
                    save_previous_trial(result, next_outcome,unix,float(event[16:20]))
                #start documentation the new trial
                clear_variables("trials",unix)                                #reset trial related values
                lickwithhold_time = round(unix - last_lick_time,2) + 1        # calculate how long the mouse really withholds its licks, add 1ms to avoid false negatives due to rounding
                last_trial_time = round(unix - TRIAL_START)
                compare_variable = min(lickwithhold_time,last_trial_time)     # in case of no licks compare with beginning of last trial
                if (float(event[16:20]) <= compare_variable):                 # evaluate if the program made a mistake and overlooked a lick during the lickwithold time
                    LICKS_WITHIN_LICKWITHHOLD_TIME = None                     # program works correct
                else:
                    LICKS_WITHIN_LICKWITHHOLD_TIME = "yes"                    # program overlooked a lick
                LICKWITHHOLD_TIME = float(event[16:20])
                TRIAL_START = unix
                result = int(event[57:])                                      #outcome of the trial
                STIMULUS = event[21:53]                                       #save stimulus, this might be overwritten later due to bad textfiles
                NOTES = event[54:]                                            #NOTES saves the GO-code we use for the trial outcomes
                check_result(result,event)
                VIDEO_NOTES = VIDEO_NOTES + event[54:]+ "\n"                  #saves all trial outcomes of a session for the video table
                if FIXATION == "fix":
                    HEADFIX_DURATION = unix - headfix_start                   #duration how long the mouse is headfixed at this timepoint
                else:
                    HEADFIX_DURATION = None                                   #no headfix time on no fix trials
                reaction_time_start = unix                                    #tracks the beginning of the reaction time variable
                reaction_time_started = True                                  #tracks the beginning of the reaction time variable
                trial_in_session_counter += 1                                 #counts the number of trials in the headfix session
                TRIAL_IN_SESSION = trial_in_session_counter                   #variable for the DB that saves the current trial in session


            if "lick:" in event:
                lick_counter_entries += 1                                     #counts licks for the entries table, so all the licks that happen during the mouse is in the chamber
                if FIXATION == "fix":
                    lick_counter_headfix += 1                                 #seperately keep track of licks made under headfixation
                # other licks than first
                if water_available == True:
                    # document if mouse got water
                    if started == False:
                        ENTRY_TYPE = "pass"                                   #mouse has probably passed the RFID reader because the lick occurs after exit
                    water_available = False                                   #assume that mouse drank all water after one lick
                    REWARD_TIME = unix                                        #save the time when the mouse got the reward for the rewards table
                    saveToDatabase("rewards")                                 #save to DB that mouse got a reward
                if reaction_time_started == False:                            #look at licks that are NOT the first lick after stimulus
                    lick_counter_trials += 1                                  #count licks
                    last_lick_time = unix                                     #keep track of the timestamp of the last recent lick
                    #determine if mouse pays attention. Important to evaluate if there are GO and NO GO trials. This can be roughly done by looking at trials before and after the current trial
                    previous_outcome_licked = True                            #marks that the mouse licked just recently. the variable will be evaluated when the next trial is saved
                    previous_outcome = 0                                      #initialize and reset the previous outcome
                # first lick after stimulus
                elif reaction_time_started == True:                           #first lick after stimulus
                    REACTION_TIME = unix - reaction_time_start                #calculate the time between stimulus and first lick
                    reaction_time_started = False                             #mark for the next lick that they are not the first after stimulus
                    lick_time_start = unix                                    #keep track of the timepoint when the mouse started licking after stimulus
                    last_lick_time = unix                                     #keep track of the timestamp of the last recent lick
                    lick_counter_trials = 1                                   #set the lick counter to 1, this is more a security than necessary
                    previous_outcome_licked = True                            #marks that the mouse licked just recently.

            # end of headfix and session
            if event == "complete":                                           #marks the time when the mouse gets released.
                had_session = True                                            #marks if a mouse exited yet acfter a session. important to process double sessions
                if headfix_start != 0:                                        #marks that a headfixation took place. headfix_start is 0 for no fix trials
                    HEADFIX_DURATION_ENTRIES = unix - headfix_start           #finalyze the variable for the DB and entries table. Displays how long the mouse was headfixed during this session
                    LICKS_WHILE_HEADFIXED = lick_counter_headfix              #finalyze the lick counts while headfixed
                else:
                    HEADFIX_DURATION_ENTRIES = None
                    LICKS_WHILE_HEADFIXED = None                              #no headfixation, no licks under headfixation

                if real_trial == True:
                    next_outcome = 0                                          #reset variable
                    save_previous_trial(result,next_outcome,unix,0)           #save the last trial
                    saveToDatabase("videos")                                  #save video information
                    VIDEO_NOTES = ""                                          #reset variable
                    trial_in_session_counter=0                                #reset variable
                    real_trial = False                                        #reset variable

            if event == "exit" and started == True:                           #double check if there was an entry before exit
                ENTRY_DURATION = unix - ENTRY_TIME                            #calculate time in chamber
                LICKS = lick_counter_entries                                  #finalyze count of licks while in chamber - entries table
                saveToDatabase("entries")                                     #save the information of the entry
                clear_variables("entries",unix)                               #reset variables related with the entries table
                had_session = False                                           #marker boolean to detect double sessions, will be used when checking bean break (check+-, check no fix) look 100 lines earlier
            if event == "exit" and started == False:
                clear_variables("entries",unix)
                had_session = False


    print("done")