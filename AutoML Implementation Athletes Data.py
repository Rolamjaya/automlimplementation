# Databricks notebook source
# MAGIC %md
# MAGIC # MLOps Assignment 3 - AutoML Implementation

# COMMAND ----------

import pandas as pd
import numpy as np

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Ingestion and Transformation
# MAGIC Data ingestion and transformation is done using pandas dataframe environment

# COMMAND ----------

df = pd.read_csv("/Workspace/Repos/rolamjayahs@gmail.com/AutoML_implementation_databricks/athletes.csv")

# COMMAND ----------

def data_transformation(df: pd.DataFrame):
    import pandas as pd
    import numpy as np
    
    df = df.dropna(subset=['region','age','weight','height','gender','eat','train','background','experience','schedule','howlong','deadlift','candj','snatch','backsq']) #removing NaNs from parameters of interest
    df = df.drop(columns=['affiliate','team','name','athlete_id','fran','helen','grace','filthy50','fgonebad','run400','run5k','pullups','train']) #removing paramters not of interest + less popular events
    
    #removing problematic entries
    df = df[df['weight'] < 1500] #removes two anomolous weight entries of 1,750 and 2,113
    df = df[df['gender']!='--'] #removes 9 non-male/female gender entries due to small sample size
    df = df[df['age']>=18] #only considering adults
    df = df[(df['height']<96)&(df['height']>48)]#selects people between 4 and 8 feet

    #no lifts above world recording holding lifts were included
    df = df[(df['deadlift']>0)&(df['deadlift']<=1105)|((df['gender']=='Female')&(df['deadlift']<=636))] #removes negative deadlift weights and deadlifts above the current world record
    df = df[(df['candj']>0)&(df['candj']<=395)]#|((df['gender']=='Female')&(df['candj']<=265))] #removes negative clean and jerk value and reported weights above the current world record
    df = df[(df['snatch']>0)&(df['snatch']<=496)]#|((df['gender']=='Female')&(df['snatch']<=341))] #removes weights above the current world record
    df = df[(df['backsq']>0)&(df['backsq']<=1069)]#|((df['gender']=='Female')&(df['backsq']<=615))] #removes weights over current world record

    #get rid of declines to answer as only response
    decline_dict = {'Decline to answer|':np.nan}
    df = df.replace(decline_dict)
    df = df.dropna(subset=['background','experience','schedule','howlong','eat'])
    
    #encoding background data

    #encoding background questions
    df['rec'] = np.where(df['background'].str.contains('I regularly play recreational sports'), 1, 0)
    df['high_school'] = np.where(df['background'].str.contains('I played youth or high school level sports'), 1, 0)
    df['college'] = np.where(df['background'].str.contains('I played college sports'), 1, 0)
    df['pro'] = np.where(df['background'].str.contains('I played professional sports'), 1, 0)
    df['no_background'] = np.where(df['background'].str.contains('I have no athletic background besides CrossFit'), 1, 0)

    #delete nonsense answers
    df = df[~(((df['high_school']==1)|(df['college']==1)|(df['pro']==1)|(df['rec']==1))&(df['no_background']==1))] #you can't have no background and also a background
    
    #encoding experience questions

    #create encoded columns for experience reponse
    df['exp_coach'] = np.where(df['experience'].str.contains('I began CrossFit with a coach'),1,0)
    df['exp_alone'] = np.where(df['experience'].str.contains('I began CrossFit by trying it alone'),1,0)
    df['exp_courses'] = np.where(df['experience'].str.contains('I have attended one or more specialty courses'),1,0)
    df['life_changing'] = np.where(df['experience'].str.contains('I have had a life changing experience due to CrossFit'),1,0)
    df['exp_trainer'] = np.where(df['experience'].str.contains('I train other people'),1,0)
    df['exp_level1'] = np.where(df['experience'].str.contains('I have completed the CrossFit Level 1 certificate course'),1,0)

    #delete nonsense answers
    df = df[~((df['exp_coach']==1)&(df['exp_alone']==1))] #you can't start alone and with a coach

    #creating no response option for coaching start
    df['exp_start_nr'] = np.where(((df['exp_coach']==0)&(df['exp_alone']==0)),1,0)

    #other options are assumed to be 0 if not explicitly selected
    
    #creating encoded columns with schedule data
    df['rest_plus'] = np.where(df['schedule'].str.contains('I typically rest 4 or more days per month'),1,0)
    df['rest_minus'] = np.where(df['schedule'].str.contains('I typically rest fewer than 4 days per month'),1,0)
    df['rest_sched'] = np.where(df['schedule'].str.contains('I strictly schedule my rest days'),1,0)

    df['sched_0extra'] = np.where(df['schedule'].str.contains('I usually only do 1 workout a day'),1,0)
    df['sched_1extra'] = np.where(df['schedule'].str.contains('I do multiple workouts in a day 1x a week'),1,0)
    df['sched_2extra'] = np.where(df['schedule'].str.contains('I do multiple workouts in a day 2x a week'),1,0)
    df['sched_3extra'] = np.where(df['schedule'].str.contains('I do multiple workouts in a day 3\+ times a week'),1,0)

    #removing/correcting problematic responses
    df = df[~((df['rest_plus']==1)&(df['rest_minus']==1))] #you can't have both more than and less than 4 rest days/month

    #points are only assigned for the highest extra workout value (3x only vs. 3x and 2x and 1x if multi selected)
    df['sched_0extra'] = np.where((df['sched_3extra']==1),0,df['sched_0extra'])
    df['sched_1extra'] = np.where((df['sched_3extra']==1),0,df['sched_1extra'])
    df['sched_2extra'] = np.where((df['sched_3extra']==1),0,df['sched_2extra'])
    df['sched_0extra'] = np.where((df['sched_2extra']==1),0,df['sched_0extra'])
    df['sched_1extra'] = np.where((df['sched_2extra']==1),0,df['sched_1extra'])
    df['sched_0extra'] = np.where((df['sched_1extra']==1),0,df['sched_0extra'])

    #adding no response columns
    df['sched_nr'] = np.where(((df['sched_0extra']==0)&(df['sched_1extra']==0)&(df['sched_2extra']==0)&(df['sched_3extra']==0)),1,0)
    df['rest_nr'] = np.where(((df['rest_plus']==0)&(df['rest_minus']==0)),1,0)
    #schedling rest days is assumed to be 0 if not explicitly selected
    
    # encoding howlong (crossfit lifetime)
    df['exp_1to2yrs'] = np.where((df['howlong'].str.contains('1-2 years')),1,0)
    df['exp_2to4yrs'] = np.where((df['howlong'].str.contains('2-4 years')),1,0)
    df['exp_4plus'] = np.where((df['howlong'].str.contains('4\+ years')),1,0)
    df['exp_6to12mo'] = np.where((df['howlong'].str.contains('6-12 months')),1,0)
    df['exp_lt6mo'] = np.where((df['howlong'].str.contains('Less than 6 months')),1,0)

    #keeping only higest repsonse
    df['exp_lt6mo'] = np.where((df['exp_4plus']==1),0,df['exp_lt6mo'])
    df['exp_6to12mo'] = np.where((df['exp_4plus']==1),0,df['exp_6to12mo'])
    df['exp_1to2yrs'] = np.where((df['exp_4plus']==1),0,df['exp_1to2yrs'])
    df['exp_2to4yrs'] = np.where((df['exp_4plus']==1),0,df['exp_2to4yrs'])
    df['exp_lt6mo'] = np.where((df['exp_2to4yrs']==1),0,df['exp_lt6mo'])
    df['exp_6to12mo'] = np.where((df['exp_2to4yrs']==1),0,df['exp_6to12mo'])
    df['exp_1to2yrs'] = np.where((df['exp_2to4yrs']==1),0,df['exp_1to2yrs'])
    df['exp_lt6mo'] = np.where((df['exp_1to2yrs']==1),0,df['exp_lt6mo'])
    df['exp_6to12mo'] = np.where((df['exp_1to2yrs']==1),0,df['exp_6to12mo'])
    df['exp_lt6mo'] = np.where((df['exp_6to12mo']==1),0,df['exp_lt6mo'])
    
    #encoding dietary preferences
    df['eat_conv'] = np.where((df['eat'].str.contains('I eat whatever is convenient')),1,0)
    df['eat_cheat']= np.where((df['eat'].str.contains('I eat 1-3 full cheat meals per week')),1,0)
    df['eat_quality']= np.where((df['eat'].str.contains('I eat quality foods but don\'t measure the amount')),1,0)
    df['eat_paleo']= np.where((df['eat'].str.contains('I eat strict Paleo')),1,0)
    df['eat_cheat']= np.where((df['eat'].str.contains('I eat 1-3 full cheat meals per week')),1,0)
    df['eat_weigh'] = np.where((df['eat'].str.contains('I weigh and measure my food')),1,0)
    
    #encoding location as US vs non-US
    US_regions = ['Southern California', 'North East', 'North Central','South East', 'South Central', 'South West', 'Mid Atlantic','Northern California','Central East', 'North West']
    df['US'] = np.where((df['region'].isin(US_regions)),1,0)
    
    #encoding gender
    df['gender_'] = np.where(df['gender']=='Male',1,0)
    
    df['norm_dl'] = df['deadlift']/df['weight']
    df['norm_j'] = df['candj']/df['weight']
    df['norm_s'] = df['snatch']/df['weight']
    df['norm_bs'] = df['backsq']/df['weight']
    
    df['BMI'] = df['weight']*0.453592/np.square(df['height']*0.0254)

    df['total_lift'] = df['norm_dl']+df['norm_j']+df['norm_s']+df['norm_bs']

    df = df[(df['BMI']>=17)&(df['BMI']<=50)] #considers only not underweight - morbidly obese competitors
    
    df = df.drop(columns=['region','height','weight','candj','snatch','deadlift','norm_bs', 'norm_dl', 'norm_j', 'norm_s','backsq','eat','background','experience','schedule','howlong','gender'])
    
    return df

# COMMAND ----------

df_transformed = data_transformation(df)

# COMMAND ----------

from sklearn.model_selection import train_test_split
 
train_pdf, test_pdf = train_test_split(df_transformed, test_size=0.2, random_state=42)
display(train_pdf)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Training
# MAGIC - The following command starts an AutoML run. We provide "total_lift" as the targetl coloumn stated in the target_col argument.
# MAGIC - When the run completes, we can follow the link to the best trial notebook to examine the training code. 
# MAGIC - This notebook also includes a feature importance plot.

# COMMAND ----------

from databricks import automl
summary = automl.regress(train_pdf, target_col="total_lift", timeout_minutes=30)

# COMMAND ----------

# MAGIC %md
# MAGIC The following command displays information about the AutoML output.

# COMMAND ----------

help(summary)

# COMMAND ----------


