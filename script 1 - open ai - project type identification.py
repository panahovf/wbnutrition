# In[]
# Author: Farhad Panahov
# Purpose: World Bank test
# Date: Jan 24, 2025

# NOTE: You will need to input your OPEN AI API KEY
# for variable: 'client'





# In[]
# packages
import os
import numpy as np
import pandas as pd
from openai import OpenAI
import json
import re





# In[]
# set directory
directory = r'C:\Users\panah\OneDrive\Desktop\Work\0 - Random\1 - WB - test - Africa Food and Nutrition Security'
os.chdir(directory)
del directory





# In[]
# load data
df_projects = pd.read_excel('1 - input/Data STC Files (FNS-Raw Dataset).xlsx', skiprows=2)




    
# In[]
# description: There are over 800 project lines. Each has name and description
# Aim is to create a 'type' that consolidated the types of the projects into a handful types

# This section uses Open AI to ask GPT4o model to classify porjects based on their
# project names and descriptions

# Data is broken in chunks to make sure GPT4o gives full responses

#----------
# enter you API key
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
# process each chunk through chatgpt separately
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
# delete
del client, chunk_size, chunks
del chunk, temp_dataformatted, existing_groups_prompt, temp_prompt, temp_response
del i, response_message, json_blocks, block, partial_mapping





# In[]
# description: there are still 270 types.
# consolidating them into ~10 types

# Continue to refine groupings. Use the 270 type descriptions and group them together

#----------
# enter you API key
client = OpenAI(api_key="ENTER YOUR API KEY")

# initialize variables# 
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

# delete
del client, unique_categories, temp_prompt, temp_response, response_message


#----------
print(len(set(combined_categories_mapping.values())))
#11




# In[]
# mapping categories to the dataset

#----------
# view types
print(set(combined_categories_mapping.values()))
# {'Infrastructure and Connectivity', 'Economic Development and Employment', 
#  'Education and Child Development', 'Climate Resilience and Sustainable Practices',
#  'Water and Agriculture Management', 'Disaster Risk Management and Resilience',
#  'Agriculture and Food Security', 'Social Protection and Resilience', 
#  'Community Development and Social Services', 'Water Management and Services', 
#  'Healthcare and Nutrition'}

# map types
df_projects['type1'] = df_projects['Project Display Name'].map(existing_group_mapping)
df_projects['type2'] = df_projects['type1'].map(combined_categories_mapping)


#----------
# unmapped
print(df_projects.loc[df_projects['type2'].isna(), 'Project Display Name'].unique())
print(df_projects.loc[df_projects['type2'].isna(), 'Project Development Objective Description'].unique())

# ['Crecer Sano: Guatemala Nutrition and Health  Project'
#  'Smallholder Irrigated Agriculture and Market Access Project- IRRIGA  1'
#  'Burkina Faso Livestock Resilience and Competitiveness  Project']

# ['The Project Development Objective (PDO) is to (i) improve selected practices, services and behaviors known to be key determinants of chronic malnutrition (with an emphasis on the first 1,000 days of life), and (ii) respond to the threat posed by COVID-19, in selected intervention areas.'
#  "The proposed Project Development Objective (PDO) is to improve smallholder agriculture productivity and market access in the project areas developed with irrigation and provide immediate and effective response to an eligible crisis or emergency.     The Program objective of the Series of Projects (SOP) is to increase farmers' productivity and improve rural livelihoods through increased access to irrigation and markets."
#  'To improve the productivity, commercialization, and resilience of key sedentary livestock production systems for targeted beneficiaries in Project areas']

# map manually
df_projects.loc[df_projects_clean['Project Display Name'] == 'Crecer Sano: Guatemala Nutrition and Health  Project', 'type2'] = "Healthcare and Nutrition"
df_projects.loc[df_projects_clean['Project Display Name'] == 'Smallholder Irrigated Agriculture and Market Access Project- IRRIGA  1', 'type2'] = "Water and Agriculture Management"
df_projects.loc[df_projects_clean['Project Display Name'] == 'Burkina Faso Livestock Resilience and Competitiveness  Project', 'type2'] = "Agriculture and Food Security"

#----------
# delete
openai_data = {
    "First query responses": all_responses,
    "First query groups": existing_group_mapping,
    "Second query groups": combined_categories_mapping}

del existing_group_mapping, combined_categories_mapping, all_responses





# In[]
# export
df_projects.to_excel('2 - output/script 1 - projects - types identified.xlsx', index=False)

with open('2 - output/script 1 - open ai outputs.json', 'w') as json_file:
    json.dump(openai_data, json_file)