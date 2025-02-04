# In[]
# Author: Farhad Panahov
# Purpose: Sample code

# NOTE: In SECTION 1 you will need to input your OPEN AI API KEY for variable: 'client'.
# For this step you have to have paid  subscription, and running will cost tokens/$
# To skip SECTION 1 --- Load the environment data and start directly at the SECTION 2





# In[]
# packages
import os
import numpy as np
import pandas as pd
import json
import re
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI





# In[]
# set directory
directory = r'INPUT YOUR DIRECTORY'
os.chdir(directory)
del directory





# In[]
# load data
df_projects = pd.read_excel('raw datafile.xlsx', skiprows=2)




   
# In[]
##################################################################################################
##################################################################################################
##################################################################################################
##################### SECTION 1: USING OPEN AI ###################################################
##################################################################################################
##################################################################################################
##################################################################################################

# Description: There are over 800 project lines. Each has a name and a description
# Aim is to create a 'type' that consolidated the projects into a handful types

# This section uses Open AI to ask GPT4o model to classify porjects based on their
# project names and descriptions

# Data is broken in chunks to make sure GPT4o gives full responses





# In[]
# Description: run an OpenAI querry to group projects into 'type's

#----------
# Enter you API key
client = OpenAI(api_key="ENTER YOUR API KEY")

# initialize variables
existing_group_mapping = {}  # This will hold all groupings created
chunk_size = 50  # Number of projects per chunk
all_responses = []  # Collect all responses

# Split the data into chunks
chunks = [
    df_projects[['Project Display Name', 'Project Development Objective Description']]
    .iloc[i:i + chunk_size]
    .to_dict(orient='records')
    for i in range(0, len(df_projects), chunk_size)
]


#----------
# Process each chunk through chatgpt separately
for i, chunk in enumerate(chunks):
    print(f"Processing chunk {i + 1} of {len(chunks)}...")

    # Format the current chunk as JSON
    temp_dataformatted = json.dumps(chunk, indent=4)

    # Include existing groups in the prompt
    existing_groups_prompt = json.dumps(existing_group_mapping, indent=4) if existing_group_mapping else "{}"

    # Create the prompt
    temp_prompt = f"""
    You are an AI tasked with grouping projects based on their development objectives. Below is a list of projects with their descriptions:

    {temp_dataformatted}

    Existing groupings are provided as a reference. Use them as much as possible to maintain consistency:
    {existing_groups_prompt}

    Please group the projects into categories based on their objectives. Limit yourself to at most 10 groupings. Return a JSON object mapping "Project Display Name" to its respective group. The JSON should look like this:
    {{
        "Project A": "Irrigation Projects",
        "Project B": "Healthcare Projects"
    }}
    """

    # Send the request
    temp_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": temp_prompt}
        ],
        temperature=1.0
    )

    # Extract response
    all_responses.append(temp_response.choices[0].message.content)


#----------
# Parse all responses after the loop
for i, response_message in enumerate(all_responses):
    try:
        # Extract JSON from each response
        json_blocks = re.findall(r'{.*}', response_message, re.DOTALL)
        for block in json_blocks:
            partial_mapping = json.loads(block)
            existing_group_mapping.update(partial_mapping)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in response {i + 1}: {e}")


#----------
print(len(set(existing_group_mapping.values())))
# 270


#----------
# Delete
del client, chunk_size, chunks
del chunk, temp_dataformatted, existing_groups_prompt, temp_prompt, temp_response
del i, response_message, json_blocks, block, partial_mapping





# In[]
# Description: there are still 270 types.
# Consolidating them into ~10 types

# Continue to refine groupings. Use the 270 type descriptions and group them together

#----------
# enter you API key
client = OpenAI(api_key="ENTER YOUR API KEY")

# Initialize variables 
unique_categories = set(existing_group_mapping.values())  # Extract unique categories from the dictionary values
unique_categories = "\n".join(unique_categories) # Format the unique categories as a string

# Prepare the prompt
temp_prompt = f"""
You are an AI tasked with consolidating project categories into broader groups. Below is a list of 270 unique project categories:

{unique_categories}

Please consolidate these categories into at most 10 broader categories. Return the result as a JSON object mapping the original category to the broader category. The JSON should look like this:
{{
    "Original Category A": "Broad Category 1",
    "Original Category B": "Broad Category 2",
    ...
}}
"""

# Run the query through OpenAI
temp_response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": temp_prompt}
    ],
    temperature=1.0)

# Extract and parse the response
response_message = temp_response.choices[0].message.content

# Parse the JSON object
try:
    combined_categories_mapping = json.loads(re.search(r'{.*}', response_message, re.DOTALL).group(0))
    print("Mapping successful!")
except json.JSONDecodeError as e:
    print("Error parsing JSON:", e)
    combined_categories_mapping = {}


#----------
print(len(set(combined_categories_mapping.values())))
#11


#----------
# Delete
del client, unique_categories, temp_prompt, temp_response, response_message





# In[]
# Description: mapping categories to the dataset

#----------
# View types
print(set(combined_categories_mapping.values()))
# {'Infrastructure and Connectivity', 'Economic Development and Employment', 
#  'Education and Child Development', 'Climate Resilience and Sustainable Practices',
#  'Water and Agriculture Management', 'Disaster Risk Management and Resilience',
#  'Agriculture and Food Security', 'Social Protection and Resilience', 
#  'Community Development and Social Services', 'Water Management and Services', 
#  'Healthcare and Nutrition'}

# Map types to the data --- both broad (270 types) and narrow version (11 types)
df_projects['type_broad'] = df_projects['Project Display Name'].map(existing_group_mapping)
df_projects['type_narrow'] = df_projects['type_broad'].map(combined_categories_mapping)


#----------
# Unmapped projects
print(df_projects.loc[df_projects['type_narrow'].isna(), 'Project Display Name'].unique())
print(df_projects.loc[df_projects['type_narrow'].isna(), 'Project Development Objective Description'].unique())

# ['Crecer Sano: Guatemala Nutrition and Health  Project'
#  'Smallholder Irrigated Agriculture and Market Access Project- IRRIGA  1'
#  'Burkina Faso Livestock Resilience and Competitiveness  Project']

# ['The Project Development Objective (PDO) is to (i) improve selected practices, services and behaviors known to be key determinants of chronic malnutrition (with an emphasis on the first 1,000 days of life), and (ii) respond to the threat posed by COVID-19, in selected intervention areas.'
#  "The proposed Project Development Objective (PDO) is to improve smallholder agriculture productivity and market access in the project areas developed with irrigation and provide immediate and effective response to an eligible crisis or emergency.     The Program objective of the Series of Projects (SOP) is to increase farmers' productivity and improve rural livelihoods through increased access to irrigation and markets."
#  'To improve the productivity, commercialization, and resilience of key sedentary livestock production systems for targeted beneficiaries in Project areas']

# Map manually
df_projects.loc[df_projects_clean['Project Display Name'] == 'Crecer Sano: Guatemala Nutrition and Health  Project', 'type_narrow'] = "Healthcare and Nutrition"
df_projects.loc[df_projects_clean['Project Display Name'] == 'Smallholder Irrigated Agriculture and Market Access Project- IRRIGA  1', 'type_narrow'] = "Water and Agriculture Management"
df_projects.loc[df_projects_clean['Project Display Name'] == 'Burkina Faso Livestock Resilience and Competitiveness  Project', 'type_narrow'] = "Agriculture and Food Security"


#----------
# Save OpenAI result into a single dictionary
openai_data = {
    "First query responses": all_responses,
    "First query groups": existing_group_mapping,
    "Second query groups": combined_categories_mapping}


#----------
# Delete
del existing_group_mapping, combined_categories_mapping, all_responses





# In[]
##################################################################################################
##################################################################################################
##################################################################################################
##################### SECTION 2: DATA ANALYSIS ###################################################
##################################################################################################
##################################################################################################
##################################################################################################

# Description: View and clean data, and prepare for visualizations
# Focus is on Africa





# In[]
# Description: continue to combine the types to fewer groups

# Combine types manually
print(df_projects['type_narrow'].unique())
# ['Water and Agriculture Management'
#  'Community Development and Social Services'
#  'Climate Resilience and Sustainable Practices' 'Healthcare and Nutrition'
#  'Infrastructure and Connectivity' 'Social Protection and Resilience'
#  'Water Management and Services' 'Agriculture and Food Security'
#  'Disaster Risk Management and Resilience'
#  'Economic Development and Employment' 'Education and Child Development']

df_projects['type_compressed'] = ''

df_projects.loc[df_projects['type_narrow'].isin(['Water and Agriculture Management',
                                          'Water Management and Services']), 'type_compressed'] = 'Water and Agriculture Management'

df_projects.loc[df_projects['type_narrow'].isin(['Community Development and Social Services',
                                          'Economic Development and Employment',
                                          'Infrastructure and Connectivity']), 'type_compressed'] = 'Economic, Social and Community Development'

df_projects.loc[df_projects['type_narrow'].isin(['Climate Resilience and Sustainable Practices',
                                          'Disaster Risk Management and Resilience']), 'type_compressed'] = 'Resilience, Sustainable Practices and Disaster Risk Management'

df_projects.loc[df_projects['type_narrow'].isin(['Community Development and Social Services',
                                          'Economic Development and Employment']), 'type_compressed'] = 'Economic, Social and Community Development'

df_projects.loc[df_projects['type_narrow'].isin(['Social Protection and Resilience',
                                          'Education and Child Development']), 'type_compressed'] = 'Social Protection, Education and Child Development'

df_projects.loc[df_projects['type_narrow'].isin(['Healthcare and Nutrition']), 'type_compressed'] = 'Healthcare and Nutrition'

df_projects.loc[df_projects['type_narrow'].isin(['Agriculture and Food Security']), 'type_compressed'] = 'Agriculture and Food Security'




# In[]
# Description: review data

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
#        'Unnamed: 37', 'type_broad', 'type_narrow'],
#       dtype='object')


#----------
# remove empty columns --- NAN columns
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
# Convert all columns to lower case
df_projects.columns = df_projects.columns.str.lower()
   
 
    
 
    
# In[]
# Description: subset data

#----------
# Subset to Africa only
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


#----------
# Delete
del african_countries    




       
# In[]
# Description: clean data --- time columns --- convert to datetime format

#----------
# Get list of columns to edit
temp_date_columns = ['approval date of the project',
 'closing date of the project',
 'archive date of the isr/icr',
 'baseline date of the indicator',
 'progress date of the indicator',
 'target date of the indicator']
    
# Run loop to convert to datetime
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

    # Step 4: convert NaN to NaT
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
# Delete
del temp_col, temp_date_columns, temp_datetimerows, temp_yyyymmddrows

    
    
    
   
# In[]
# Description: clean data --- create new variables

#----------
# INDICATOR: create project completion to target --- indicator
# first convert columns to numeric format and then compute progress rate
df_projects['target value of the indicator'] = pd.to_numeric(df_projects['target value of the indicator'], errors='coerce')
df_projects['baseline value of the indicator'] = pd.to_numeric(df_projects['baseline value of the indicator'], errors='coerce')
df_projects['progress value'] = pd.to_numeric(df_projects['progress value'], errors='coerce')

df_projects['completion_rate_indicator'] = df_projects['progress value'] / df_projects['target value of the indicator']


#----------
# DATE: create project completion to target - date
# (1) days overall | (2) days passed from start | (3) days passed as share of total days
df_projects['project_lenght_days'] = df_projects['target date of the indicator'] - df_projects['baseline date of the indicator']
df_projects['project_lenght_lastmeasure'] = df_projects['progress date of the indicator'] - df_projects['baseline date of the indicator']
df_projects['completion_rate_time'] = df_projects['project_lenght_lastmeasure']/df_projects['project_lenght_days']


#----------
# Remove rows with NANs in specific columns
columns_to_check = ['net commitment amount - total', 
                    'baseline value of the indicator', 'progress value',
                    'baseline date of the indicator','progress date of the indicator',
                    'target value of the indicator', 'target date of the indicator']

df_projects = df_projects.dropna(subset=columns_to_check, axis=0)

print(len(df_projects['bank assigned project unique id assigned to a specific operation'].unique()), 'unique projects')
# 203 unique projects


#----------
print(df_projects['completion_rate_indicator'].describe())
# count      372.000000
# mean       268.593316
# std       3199.954539
# min          0.000000
# 25%         16.822917
# 50%         69.701019
# 75%        125.302083
# max      61740.000000
# Name: completion_rate_indicator, dtype: float64

print(df_projects['completion_rate_time'].describe())
# count    374.000000
# mean       0.681990
# std        0.411770
# min       -0.035426
# 25%        0.352685
# 50%        0.680654
# 75%        0.943453
# max        3.023810
# Name: completion_rate_time, dtype: float64


#----------
# PERIOD: finished projects, ongoing, and future
# add categories based on approval and closing dates
df_projects['status'] = ''
df_projects.loc[df_projects['closing date of the project'] <= '2024-12-31', 'status'] = "Past"
df_projects.loc[(df_projects['closing date of the project'] > '2024-12-31') & (df_projects['approval date of the project'] <= '2024-12-31'), 'status'] = "Current"
df_projects.loc[df_projects['approval date of the project'] > '2024-12-31', 'status'] = "Future"


#----------
# Delete
del columns_to_check





# In[]
##################################################################################################
##################################################################################################
##################################################################################################
##################### SECTION 3: PLOTS ###########################################################
##################################################################################################
##################################################################################################
##################################################################################################

#----------
# Description: prepare for plotting

# 1 --- style
plt.style.use('tableau-colorblind10')

# 2 --- convert relevant columns to numeric
df_projects['project approval fiscal year'] = pd.to_numeric(df_projects['project approval fiscal year'], errors='coerce')
df_projects['net commitment amount - total'] = pd.to_numeric(df_projects['net commitment amount - total'], errors='coerce')
df_projects['approval date of the project'] = pd.to_datetime(df_projects['approval date of the project'], errors='coerce')
df_projects['closing date of the project'] = pd.to_datetime(df_projects['closing date of the project'], errors='coerce')





# In[]
# Description: PLOT 1: VALUE AND COUNT OF PROJECTS OF TIME

# Create a copy of the original data
df_projects_a = df_projects.copy()

# Group by approval year and aggregate data
annual_data_a = df_projects_a.groupby('project approval fiscal year').agg(
    number_of_projects=('bank assigned project unique id assigned to a specific operation', 'count'),
    net_commitment=('net commitment amount - total', 'sum')
).reset_index()

# Convert net commitment to billions for readability
annual_data_a['net_commitment_billions'] = annual_data_a['net_commitment'] / 1e9

# Plot the chart
fig, ax1 = plt.subplots(figsize=(6, 6))

# Bar plot: number of projects (orange)
ax1.bar(
    annual_data_a['project approval fiscal year'],
    annual_data_a['number_of_projects'],
    color='orange',
    alpha=0.8,
    label="Number of Projects"  #legend
)

ax1.set_xlabel("Approval year")  # Remove x-axis title
ax1.set_ylabel("Count")  # Remove y-axis title for the left axis
ax1.tick_params(axis='y', labelcolor='black', labelsize=12)  # Set font size for left y-axis
ax1.tick_params(axis='x', labelsize=12)  # Set font size for x-axis
ax1.grid(True, which='major', axis='y', linestyle='--', alpha=0.6)  # Add grid lines for the first y-axis

# Set y-axis limits for the first axis
ax1.set_ylim(0, 85)

# Line plot: net commitment (dark blue, scaled to billions)
ax2 = ax1.twinx()
ax2.plot(
    annual_data_a['project approval fiscal year'],
    annual_data_a['net_commitment_billions'],
    color='darkblue',
    marker='o',
    label="Total approved funding"  # legend
)

ax2.set_ylabel("Billions USD")  # Remove y-axis title for the right axis
ax2.tick_params(axis='y', labelcolor='black', labelsize=12)  # Set font size for right y-axis
ax2.grid(False)  # Remove grid lines for the second axis

# Set y-axis limits for the second axis to start at zero
ax2.set_ylim(0, 17)

# chart title
fig.suptitle("Approved Projects and Funding", fontsize=16, fontweight='bold')
plt.subplots_adjust(top=0.9)  # Adjusts space for the title

# Show plot
plt.show()





# In[]
# Description: PLOT 2: ANNUAL PROJECT PIPELINE
# Here we first create a new dataframe tthat counts how many projects are active based on their start and close years

# Create a copy of the original data
df_projects_b = df_projects.copy()

# Create a range of years covering the data
min_year = df_projects_b['approval date of the project'].dt.year.min()
max_year = df_projects_b['closing date of the project'].dt.year.max()
year_range = range(min_year, max_year + 1)

# Generate a dataframe to track active projects per year, stacked by status
active_projects_by_year = pd.DataFrame({'year': list(year_range)}).set_index('year')

# Dynamically track statuses from the dataset
statuses = df_projects_b['status'].unique()

# Add columns for each status dynamically
for status in statuses:
    active_projects_by_year[status] = 0

# Iterate through each project to count active projects by year and status
for _, row in df_projects_b.iterrows():
    if pd.notnull(row['approval date of the project']) and pd.notnull(row['closing date of the project']):
        start_year = row['approval date of the project'].year
        end_year = row['closing date of the project'].year
        status = row['status']
        if status in active_projects_by_year.columns:
            active_projects_by_year.loc[start_year:end_year, status] += 1

# Define specific colors for "Past" and "Future" statuses
color_mapping = {
    "Past": "darkblue",  # Mid-light blue
    "Future": "#4D4D4D",  # Mid-dark grey
}

# Apply color mapping to available statuses
colors = [color_mapping.get(status, "#a6a6a6") for status in active_projects_by_year.columns]

# Plot the stacked bar chart
fig, ax1 = plt.subplots(figsize=(6, 6))

# Plot the stacked bars
active_projects_by_year.plot(
    kind='bar',
    stacked=True,
    ax=ax1,
    width=0.8,
    color=colors,
    alpha=0.95
)

# Adjust x-axis ticks to show every 5 years and make them horizontal
ax1.set_xticks(range(0, len(active_projects_by_year.index), 5))
ax1.set_xticklabels(active_projects_by_year.index[::5], rotation=0)

# Remove x-axis and y-axis labels
ax1.set_xlabel("Year")
ax1.set_ylabel("Count")

# Set font sizes for axis tick labels
ax1.tick_params(axis='y', labelcolor='black', labelsize=12)  # Set font size for left y-axis
ax1.tick_params(axis='x', labelsize=12)  # Set font size for x-axis

# Add grid lines for the y-axis
ax1.grid(True, which='major', axis='y', linestyle='--', alpha=0.6)

# Remove extra spacing in plot
plt.tight_layout()

# chart title
fig.suptitle("Active projects", fontsize=16, fontweight='bold')
plt.subplots_adjust(top=0.9)  # Adjusts space for the title

# Show
plt.show()





# In[]
# Description: PLOT 3: PROJECTS BY TYPE

# Create a copy of the original DataFrame
df_projects_c = df_projects.copy()

# Group by type (type_compressed) to calculate total projects and average value of amounts
type_compressed_summary = df_projects_c.groupby('type_compressed').agg(
    total_projects=('bank assigned project unique id assigned to a specific operation', 'count'),
    avg_commitment=('net commitment amount - total', 'mean')
).reset_index()

# Sort by total_projects in descending order
type_compressed_summary = type_compressed_summary.sort_values(by='total_projects', ascending=False)

# Function to split type_compressed names into two lines
def split_into_two_lines(name):
    words = name.split()
    mid = len(words) // 2  # Split approximately in the middle
    if len(words) > 2:
        return ' '.join(words[:mid]) + '\n' + ' '.join(words[mid:])
    return name

# Apply the splitting function
type_compressed_summary['type_compressed'] = type_compressed_summary['type_compressed'].apply(split_into_two_lines)

# Reverse the order for the chart to display high-to-low from top to bottom
type_compressed_summary = type_compressed_summary[::-1]

# Define y positions for the bars and dots
y_positions = range(len(type_compressed_summary))

# Plot the horizontal bar chart with dot labels positioned next to the dots
fig, ax1 = plt.subplots(figsize=(6, 6))  # Taller chart for longer names

# Horizontal bar chart for total number of projects
ax1.barh(
    type_compressed_summary['type_compressed'],
    type_compressed_summary['total_projects'],
    color='#5DADE2',  # Bar color
    alpha=0.8
)
ax1.set_xlabel("Count")  # Remove x-axis title
ax1.tick_params(axis='y', labelsize=10)  # Adjust font size for bar names
ax1.grid(True, axis='x', linestyle='--', alpha=0.6)  # Add grid lines on the x-axis
ax1.tick_params(axis='x', labelsize=12)

# Adjust y-axis tick labels alignment
ax1.set_yticks(y_positions)
ax1.set_yticklabels(type_compressed_summary['type_compressed'], ha='right')

# Add a secondary axis for the dots (average commitment)
ax2 = ax1.twiny()  # Create a secondary x-axis
ax2.set_xlim(0, type_compressed_summary['avg_commitment'].max() / 1e6 * 1.1)  # Scale for the second axis
ax2.tick_params(axis='x', colors='none')  # Hide ticks and labels for the secondary axis
ax2.grid(False)  # Remove grid lines for the secondary axis

# Plot the dots for average commitment amount on the secondary axis with larger size
ax2.scatter(
    type_compressed_summary['avg_commitment'] / 1e6,  # Convert to millions
    y_positions,  # Match y-axis position
    color='darkblue',  # Dot color
    s=50  # Larger dots
)

# Add labels next to the dots (to the right and slightly higher)
for y, row in zip(y_positions, type_compressed_summary.itertuples()):
    ax2.text(
        row.avg_commitment / 1e6 + 10,  # Slightly to the right of the dot
        y,  # Y position aligned with the dot
        f"${row.avg_commitment / 1e6:.1f}M",
        va='center',
        fontsize=12,  # Larger font size for labels
        color='darkblue'  # Match label color to dot color
    )

# Remove all chart box lines (spines) for ax1 (main axis)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Remove all chart box lines (spines) for ax2 (secondary axis)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['bottom'].set_visible(False)

# Remove extra spacing in plot
plt.tight_layout()

# chart title
fig.suptitle("Projects and funding by program type", fontsize=16, fontweight='bold')
plt.subplots_adjust(top=0.9)  # Adjusts space for the title

# Show
plt.show()








