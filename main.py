import csv
import os
import pandas as pd
import numpy as np
csv.field_size_limit(1000000000)

# read csv file from data folder
df = pd.read_csv(os.path.join('data', 'state-of-the-union.csv'), skiprows=0, names=['year','text'])
# df.columns = ['year', 'text']
# see first 10 rows
temp = df.head(1)
# print 2nd column of temp
print(temp.iloc[0, 1])

