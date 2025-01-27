# In[]
# Author: Farhad Panahov
# Purpose: World Bank test
# Date: Jan 24, 2025

# NOTE: Plotting





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
df_projects = pd.read_excel('2 - output/script 2 - projects - by status.xlsx')
plt.style.use('tableau-colorblind10')



# In[]
##################################################################################################
##################### PLOT 1: ANNUAL PROJECT # AND $ #############################################
##################################################################################################

# Create a copy of the original DataFrame
df_projects_a = df_projects.copy()

# Convert relevant columns to numeric
df_projects_a['project approval fiscal year'] = pd.to_numeric(df_projects_a['project approval fiscal year'], errors='coerce')
df_projects_a['net commitment amount - total'] = pd.to_numeric(df_projects_a['net commitment amount - total'], errors='coerce')

# Group by approval year and aggregate data
annual_data_a = df_projects_a.groupby('project approval fiscal year').agg(
    number_of_projects=('bank assigned project unique id assigned to a specific operation', 'count'),
    net_commitment=('net commitment amount - total', 'sum')
).reset_index()

# Convert net commitment to billions for readability
annual_data_a['net_commitment_billions'] = annual_data_a['net_commitment'] / 1e9

# Plot the chart
fig, ax1 = plt.subplots(figsize=(6, 6))

# Bar plot for the number of projects (orange)
ax1.bar(
    annual_data_a['project approval fiscal year'],
    annual_data_a['number_of_projects'],
    color='orange',
    alpha=0.8
)
ax1.set_xlabel(None)  # Remove x-axis title
ax1.set_ylabel(None)  # Remove y-axis title for the left axis
ax1.tick_params(axis='y', labelcolor='black', labelsize=12)  # Set font size for left y-axis
ax1.tick_params(axis='x', labelsize=12)  # Set font size for x-axis
ax1.grid(True, which='major', axis='y', linestyle='--', alpha=0.6)  # Add grid lines for the first y-axis

# Remove box plot lines (spines) for the first axis
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['bottom'].set_visible(False)

# Set y-axis limits for the first axis (auto-adjust without starting at zero)
ax1.set_ylim(0, 85)

# Line plot for net commitment (dark blue, scaled to billions)
ax2 = ax1.twinx()
ax2.plot(
    annual_data_a['project approval fiscal year'],
    annual_data_a['net_commitment_billions'],
    color='darkblue',
    marker='o'
)
ax2.set_ylabel(None)  # Remove y-axis title for the right axis
ax2.tick_params(axis='y', labelcolor='black', labelsize=12)  # Set font size for right y-axis
ax2.grid(False)  # Remove grid lines for the second axis

# Remove box plot lines (spines) for the second axis
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Set y-axis limits for the second axis to start at zero
ax2.set_ylim(0, 17)

# Remove chart title
fig.suptitle(None)

plt.show()






# In[]
##################################################################################################
##################### PLOT 2: ANNUAL PROJECT PIPELINE ############################################
##################################################################################################

# Create a copy of the original DataFrame
df_projects_b = df_projects.copy()

# Convert relevant date columns to datetime format
df_projects_b['approval date of the project'] = pd.to_datetime(df_projects_b['approval date of the project'], errors='coerce')
df_projects_b['closing date of the project'] = pd.to_datetime(df_projects_b['closing date of the project'], errors='coerce')

# Create a range of years covering the data
min_year = df_projects_b['approval date of the project'].dt.year.min()
max_year = df_projects_b['closing date of the project'].dt.year.max()
year_range = range(min_year, max_year + 1)

# Generate a DataFrame to track active projects per year, stacked by status
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
ax1.set_xlabel(None)
ax1.set_ylabel(None)

# Set font sizes for axis tick labels
ax1.tick_params(axis='x', labelsize=12)
ax1.tick_params(axis='y', labelsize=12)

# Add grid lines for the y-axis
ax1.grid(True, which='major', axis='y', linestyle='--', alpha=0.6)

# Remove box plot lines (spines) for the axes
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Remove chart title and legend
fig.suptitle(None)
ax1.legend().remove()

plt.tight_layout()
plt.show()





# In[]
##################################################################################################
##################### PLOT 2: ANNUAL PROJECT PIPELINE ############################################
##################################################################################################

# Create a copy of the original DataFrame
df_projects_c = df_projects.copy()

# Ensure relevant columns are numeric for calculations
df_projects_c['net commitment amount - total'] = pd.to_numeric(df_projects_c['net commitment amount - total'], errors='coerce')

# Group by 'type2' to calculate total projects and average value of amounts
type2_summary = df_projects_c.groupby('type3').agg(
    total_projects=('bank assigned project unique id assigned to a specific operation', 'count'),
    avg_commitment=('net commitment amount - total', 'mean')
).reset_index()

# Sort by total_projects in descending order
type2_summary = type2_summary.sort_values(by='total_projects', ascending=False)

# Function to split type2 names into two lines
def split_into_two_lines(name):
    words = name.split()
    mid = len(words) // 2  # Split approximately in the middle
    if len(words) > 2:
        return ' '.join(words[:mid]) + '\n' + ' '.join(words[mid:])
    return name

# Apply the splitting function
type2_summary['type3'] = type2_summary['type3'].apply(split_into_two_lines)

# Reverse the order for the chart to display high-to-low from top to bottom
type2_summary = type2_summary[::-1]

# Define y positions for the bars and dots
y_positions = range(len(type2_summary))

# Plot the horizontal bar chart with dot labels positioned next to the dots
fig, ax1 = plt.subplots(figsize=(8, 12))  # Taller chart for longer names

# Horizontal bar chart for total number of projects
ax1.barh(
    type2_summary['type3'],
    type2_summary['total_projects'],
    color='#5DADE2',  # Bar color
    alpha=0.8
)
ax1.set_xlabel(None)  # Remove x-axis title
ax1.set_ylabel(None)  # Remove y-axis title
ax1.tick_params(axis='y', labelsize=16)  # Adjust font size for bar names
ax1.grid(True, axis='x', linestyle='--', alpha=0.6)  # Add grid lines on the x-axis
ax1.tick_params(axis='x', labelsize=16)

# Adjust y-axis tick labels alignment
ax1.set_yticks(y_positions)
ax1.set_yticklabels(type2_summary['type3'], ha='right')

# Add a secondary axis for the dots (average commitment)
ax2 = ax1.twiny()  # Create a secondary x-axis
ax2.set_xlim(0, type2_summary['avg_commitment'].max() / 1e6 * 1.1)  # Scale for the second axis
ax2.tick_params(axis='x', colors='none')  # Hide ticks and labels for the secondary axis
ax2.grid(False)  # Remove grid lines for the secondary axis

# Plot the dots for average commitment amount on the secondary axis with larger size
ax2.scatter(
    type2_summary['avg_commitment'] / 1e6,  # Convert to millions
    y_positions,  # Match y-axis position
    color='darkblue',  # Dot color
    s=100  # Larger dots
)

# Add labels next to the dots (to the right and slightly higher)
for y, row in zip(y_positions, type2_summary.itertuples()):
    ax2.text(
        row.avg_commitment / 1e6 + 10,  # Slightly to the right of the dot
        y,  # Y position aligned with the dot
        f"{row.avg_commitment / 1e6:.1f}M",
        va='center',
        fontsize=16,  # Larger font size for labels
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

# Remove chart title and unnecessary elements
fig.suptitle(None)  # Remove title
plt.tight_layout()
plt.show()





