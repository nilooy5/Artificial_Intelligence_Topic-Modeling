import csv
import os
import pandas as pd
import numpy as np
csv.field_size_limit(1000000000)

# read csv file from data folder
df = pd.read_csv(os.path.join('data', 'state-of-the-union.csv'), names=['year', 'speech'], skiprows=1)

df['speech'] = df['speech'].str.replace('\nState of the Union Address\n', '')
df['speech'] = df['speech'].str.replace('\nAddress to Joint Session of Congress \n', '')
df['speech'] = df['speech'].str.replace('\nAddress on Administration Goals (Budget Message)\n', '')
df['speech'] = df['speech'].str.replace('\nAddress on Administration Goals\n', '')
df['speech'] = df['speech'].str.replace('\nAddress to Congress \n', '')

df['president'] = df['speech']

df['president'] = df['president'].str.split('\n').str[0]
df['date'] = df['speech'].str.split('\n').str[1]

temp_date = df[df['date'] == 'Address on Administration Goals (Budget Message)']['speech'].str.split('\n').str[3]
df['date'][df['date'] == 'Address on Administration Goals (Budget Message)'] = temp_date.values[0]

# delete first 3 lines of speech
df['speech'] = df['speech'].str.split('\n').str[3:]
# make a string list
df['speech'] = df['speech'].str.join(' ')
print(df['speech'].values)