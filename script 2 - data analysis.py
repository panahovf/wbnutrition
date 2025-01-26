# In[]
# Author: Farhad Panahov
# Purpose: World Bank test
# Date: Jan 24, 2025

# NOTE: Data cleaning and analysis





# In[]
# packages
import os
import numpy as np
import pandas as pd
from openai import OpenAI
import json
import re
import matplotlib.pyplot as plt
import seaborn as sns




# In[]
# set directory
directory = r'C:\Users\panah\OneDrive\Desktop\Work\0 - Random\1 - WB - test - Africa Food and Nutrition Security'
os.chdir(directory)
del directory





# In[]
# load data
df_projects = pd.read_excel('2 - output/script 1 - projects - types identified.xlsx')





# In[]
# combine types
print(df_projects['type2'].unique())
# ['Water and Agriculture Management'
#  'Community Development and Social Services'
#  'Climate Resilience and Sustainable Practices' 'Healthcare and Nutrition'
#  'Infrastructure and Connectivity' 'Social Protection and Resilience'
#  'Water Management and Services' 'Agriculture and Food Security'
#  'Disaster Risk Management and Resilience'
#  'Economic Development and Employment' 'Education and Child Development']

df_projects['type3'] = ''

df_projects.loc[df_projects['type2'].isin(['Water and Agriculture Management',
                                          'Water Management and Services']), 'type3'] = 'Water and Agriculture Management'

df_projects.loc[df_projects['type2'].isin(['Community Development and Social Services',
                                          'Economic Development and Employment',
                                          'Infrastructure and Connectivity']), 'type3'] = 'Economic, Social and Community Development'

df_projects.loc[df_projects['type2'].isin(['Climate Resilience and Sustainable Practices',
                                          'Disaster Risk Management and Resilience']), 'type3'] = 'Resilience, Sustainable Practices and Disaster Risk Management'

df_projects.loc[df_projects['type2'].isin(['Community Development and Social Services',
                                          'Economic Development and Employment']), 'type3'] = 'Economic, Social and Community Development'

df_projects.loc[df_projects['type2'].isin(['Social Protection and Resilience',
                                          'Education and Child Development']), 'type3'] = 'Social Protection, Education and Child Development'





# In[]
# review data

#----------
print(df_projects.columns)
# Index(['Bank assigned Project Unique ID assigned to a specific operation',
#        'Project Display Name', 'Project Development Objective Description',
#        'Project Approval Fiscal Year', 'Approval date of the project',
#        'Closing date of the project', 'Region Name', 'Country Name',
#        'Lead Group Practice Name', 'Project State Name',
#        'Lending Instrument Name', 'Project Financer Name',
#        'Agreement Type Name', 'Net Commitment Amount - Total',
#        'Net Commitment Amount - IBRD', 'Net Commitment Amount - IDA',
#        'Net Commitment Amount - Others', 'Name of the indicator',
#        'Unit of the measure of the indicator', 'Archive Date of the ISR/ICR',
#        'Sequence number of the ISR or state if its an ICR',
#        'Resporting fiscal year of the ISR/ICR',
#        'Baseline date of the indicator', 'Baseline value of the indicator',
#        'Progress date of the Indicator', 'Progress Value',
#        'Target Date of the indicator', 'Target Value of the indicator',
#        'Unnamed: 28', 'Unnamed: 29', 'Unnamed: 30', 'Unnamed: 31',
#        'Unnamed: 32', 'Upper middle income', 'Unnamed: 34', 'No', 'MNA',
#        'Unnamed: 37', 'type1', 'type2'],
#       dtype='object')


print(df_projects.head(1))
#   Bank assigned Project Unique ID assigned to a specific operation  \
# 0                                            P086592                 

#                                 Project Display Name  \
# 0  Second Irrigation and Drainage Improvement Pro...   

#            Project Development Objective Description  \
# 0  The Project development objective is to improv...   

#    Project Approval Fiscal Year Approval date of the project  \
# 0                          2013                   2013-06-27   

#   Closing date of the project Region Name Country Name  \
# 0                  2025-06-30         ECA   Kazakhstan   

#   Lead Group Practice Name Project State Name  ... Unnamed: 30 Unnamed: 31  \
# 0                      WAT             Active  ...         NaN         NaN   

#   Unnamed: 32  Upper middle income  Unnamed: 34  No  MNA Unnamed: 37  \
# 0         NaN                  NaN          NaN NaN  NaN         NaN   

#                  type1                             type2  
# 0  Irrigation Projects  Water and Agriculture Management  

# [1 rows x 40 columns]

    
#----------
# remove NAN columns
df_projects = df_projects.dropna(axis=1, how='all')
print(df_projects.describe())

#        Project Approval Fiscal Year   Approval date of the project  \
# count                    845.000000                            845   
# mean                    2019.725444  2019-11-14 17:16:07.100591616   
# min                     2011.000000            2011-04-07 00:00:00   
# 25%                     2018.000000            2018-02-28 00:00:00   
# 50%                     2020.000000            2020-05-15 00:00:00   
# 75%                     2022.000000            2022-03-25 00:00:00   
# max                     2024.000000            2024-04-14 00:00:00   
# std                        2.764514                            NaN   

#          Closing date of the project  Net Commitment Amount - Total  \
# count                            845                   8.450000e+02   
# mean   2026-02-23 17:57:01.065088768                   1.604718e+08   
# min              2022-06-30 00:00:00                   1.950000e+06   
# 25%              2024-06-30 00:00:00                   5.000000e+07   
# 50%              2025-12-31 00:00:00                   1.000000e+08   
# 75%              2027-06-15 00:00:00                   2.000000e+08   
# max              2030-12-31 00:00:00                   8.498200e+08   
# std                              NaN                   1.615826e+08   

#        Net Commitment Amount - IBRD  Net Commitment Amount - IDA  \
# count                  8.390000e+02                 8.420000e+02   
# mean                   4.712570e+07                 1.095251e+08   
# min                    0.000000e+00                 0.000000e+00   
# 25%                    0.000000e+00                 0.000000e+00   
# 50%                    0.000000e+00                 5.500000e+07   
# 75%                    0.000000e+00                 1.500000e+08   
# max                    8.498200e+08                 7.881000e+08   
# std                    1.208364e+08                 1.464158e+08   

#        Net Commitment Amount - Others  Resporting fiscal year of the ISR/ICR  \
# count                    8.450000e+02                             840.000000   
# mean                     4.538283e+06                            2023.909524   
# min                      0.000000e+00                            2023.000000   
# 25%                      0.000000e+00                            2024.000000   
# 50%                      0.000000e+00                            2024.000000   
# 75%                      0.000000e+00                            2024.000000   
# max                      5.030000e+08                            2024.000000   
# std                      3.000810e+07                               0.287034   

#        Progress Value  
# count    8.450000e+02  
# mean     5.443179e+05  
# min      0.000000e+00  
# 25%      0.000000e+00  
# 50%      1.067900e+04  
# 75%      1.465380e+05  
# max      2.037745e+07  
# std      2.132127e+06 


    
#----------
# convert all columns to lower case
df_projects.columns = df_projects.columns.str.lower()
   
 
    
 
    
# In[]
# subset

# subset to Africa only
african_countries = [
    "Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi",
    "Cabo Verde", "Cameroon", "Central African Republic", "Chad", "Comoros",
    "Congo, Republic of", "Congo, Democratic Republic of", "Djibouti",
    "Egypt, Arab Republic of", "Equatorial Guinea", "Eritrea", "Eswatini", "Ethiopia", "Gabon",
    "Gambia, The", "Ghana", "Guinea", "Guinea-Bissau", "Côte d'Ivoire", "Kenya",
    "Lesotho", "Liberia", "Madagascar", "Malawi", "Mali", "Mauritania",
    "Mauritius", "Morocco", "Mozambique", "Namibia", "Niger", "Nigeria", "Rwanda",
    "São Tomé and Príncipe", "Senegal", "Seychelles", "Sierra Leone", "Somalia",
    "South Africa", "South Sudan", "Sudan", "Tanzania", "Togo", "Tunisia",
    "Uganda", "Zambia", "Zimbabwe"]

df_projects = df_projects[df_projects['country name'].isin(african_countries)]

# delete
del african_countries    


       
# In[]
# clean data --- time columns

#----------
# edit date columns

# get list of columns to edit
temp_date_columns = ['approval date of the project',
 'closing date of the project',
 'archive date of the isr/icr',
 'baseline date of the indicator',
 'progress date of the indicator',
 'target date of the indicator']
    
# run loop to convert to datetime
for temp_col in temp_date_columns:
    # Step 1: convert columns to strings
    df_projects[temp_col] = df_projects[temp_col].astype(str).str.strip()

    # Step 2: identify rows 
    # (1) already in date/time format
    temp_datetimerows = df_projects[temp_col].str.contains(r'^\d{4}-\d{2}-\d{2}', na=False)
    
    # (2) with the YYYYMMDD format
    temp_yyyymmddrows = df_projects[temp_col].str.contains(r'^\d{8}$', na=False)

    # Step 3: convert to datetime
    # (1) standard datetime rows
    df_projects.loc[temp_datetimerows, temp_col] = pd.to_datetime(
        df_projects.loc[temp_datetimerows, temp_col],
        errors='coerce')

    # (2) YYYYMMDD rows
    df_projects.loc[temp_yyyymmddrows, temp_col] = pd.to_datetime(
        df_projects.loc[temp_yyyymmddrows, temp_col],
        format='%Y%m%d', errors='coerce')

    # Step 5: convert NaN to NaT
    df_projects[temp_col] = pd.to_datetime(df_projects[temp_col], errors='coerce')

    # print for check
    print(type(df_projects[temp_col]))
    print(df_projects[temp_col].min())
    print(df_projects[temp_col].max())

    # [8 rows x 9 columns]
    # <class 'pandas.core.series.Series'>
    # 2011-04-07 00:00:00
    # 2024-04-14 00:00:00
    # <class 'pandas.core.series.Series'>
    # 2022-06-30 00:00:00
    # 2029-12-31 00:00:00
    # <class 'pandas.core.series.Series'>
    # 2023-03-22 00:00:00
    # 2024-06-30 00:00:00
    # <class 'pandas.core.series.Series'>
    # 2011-03-15 00:00:00
    # 2024-06-02 00:00:00
    # <class 'pandas.core.series.Series'>
    # 2017-09-29 00:00:00
    # 2024-06-27 00:00:00
    # <class 'pandas.core.series.Series'>
    # 2018-10-19 00:00:00
    # 2029-12-31 00:00:00
  
#----------
# delete
del temp_col, temp_date_columns, temp_datetimerows, temp_yyyymmddrows

    
    
    
   
# In[]
# clean data --- create new variables

df_projects_clean1 = df_projects.copy()
print(len(df_projects['bank assigned project unique id assigned to a specific operation'].unique()))
# 206 --- unique projects


#----------
# create project completion to target - indicator
df_projects_clean1['target value of the indicator'] = pd.to_numeric(df_projects_clean1['target value of the indicator'], errors='coerce')
df_projects_clean1['baseline value of the indicator'] = pd.to_numeric(df_projects_clean1['baseline value of the indicator'], errors='coerce')
df_projects_clean1['progress value'] = pd.to_numeric(df_projects_clean1['progress value'], errors='coerce')

df_projects_clean1['completion_rate_indicator'] = df_projects_clean1['progress value'] / df_projects_clean1['target value of the indicator']


#----------
# create project completion to target - date
# 1 - days overall | 2 - days passed | 3 - days passes as share of total days
df_projects_clean1['project_lenght_days'] = df_projects_clean1['target date of the indicator'] - df_projects_clean1['baseline date of the indicator']
df_projects_clean1['project_lenght_lastmeasure'] = df_projects_clean1['progress date of the indicator'] - df_projects_clean1['baseline date of the indicator']
df_projects_clean1['completion_rate_time'] = df_projects_clean1['project_lenght_lastmeasure']/df_projects_clean1['project_lenght_days']


#----------
# remove rows with NANs in specific columns
columns_to_check1 = ['net commitment amount - total', 
                    'baseline value of the indicator', 'progress value',
                    'baseline date of the indicator','progress date of the indicator',
                    'target value of the indicator', 'target date of the indicator']
df_projects_clean1 = df_projects_clean1.dropna(subset=columns_to_check1, axis=0)
print(len(df_projects_clean1['bank assigned project unique id assigned to a specific operation'].unique()))
# 203 --- unique projects


#----------
print(df_projects_clean1['completion_rate_indicator'].describe())
# count      372.000000
# mean       268.593316
# std       3199.954539
# min          0.000000
# 25%         16.822917
# 50%         69.701019
# 75%        125.302083
# max      61740.000000
# Name: completion_rate_indicator, dtype: float64


print(df_projects_clean1['completion_rate_time'].describe())
# count    374.000000
# mean       0.681990
# std        0.411770
# min       -0.035426
# 25%        0.352685
# 50%        0.680654
# 75%        0.943453
# max        3.023810
# Name: completion_rate_time, dtype: float64



# In[]
# subset

#----------
# finished projects, ongoing, and future
df_projects_clean_past = df_projects_clean1[df_projects_clean1['target date of the indicator'] <= '2024-12-31']
df_projects_clean_current = df_projects_clean1[(df_projects_clean1['target date of the indicator'] > '2024-12-31') & (df_projects_clean1['baseline date of the indicator'] <= '2024-12-31')]
df_projects_clean_future = df_projects_clean1[df_projects_clean1['baseline date of the indicator'] > '2024-12-31']


#----------
# add similar categories to original df --- based on approval and closing dates
df_projects['status'] = ''
df_projects.loc[df_projects['closing date of the project'] <= '2024-12-31', 'status'] = "Past"
df_projects.loc[(df_projects['closing date of the project'] > '2024-12-31') & (df_projects['approval date of the project'] <= '2024-12-31'), 'status'] = "Current"
df_projects.loc[df_projects['approval date of the project'] > '2024-12-31', 'status'] = "Future"


#----------
# delete
del df_projects_clean_future   # empty, no future projects




# In[]
# export
df_projects.to_excel('2 - output/script 2 - projects - by status.xlsx', index=False)
df_projects_clean_current.to_excel('2 - output/script 2 - projects - current.xlsx', index=False)
df_projects_clean_past.to_excel('2 - output/script 2 - projects - past.xlsx', index=False)









