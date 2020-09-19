'''
This code goes with my lda_topic_modeling code
I got the data for this project from Web of Science, which I accessed through my university's libary.  

Web of Science only allowed me to export 500 patents at a time. There were 3,475 for my particular search terms ('facial recognition') so I exported seven total files of patent data. This file quickly and easily combines the seven data files into one file. It also isolates and renames the columns I am interested in.

For my purposes, I am only interested in patent abstracts, countries of origin, and application dates.
'''

import pandas as pd
import glob

# The following code was taken from this website:
# https://stackoverflow.com/questions/20906474/import-multiple-csv-files-into-pandas-and-concatenate-into-one-dataframe
path = 'Downloads/wos_facial_recognition_data'
all_files = glob.glob(path + '/*.txt')

li = []

for filename in all_files:
    df = pd.read_csv(filename, sep='\t', lineterminator='\r', encoding='utf16', index_col=False, header=0)
    li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)

data = [frame[' AB '], frame['PN '], frame[' PI ']]

headers = ['abstract', 'PN', 'PI']

clean_frame = pd.concat(data, axis=1, keys=headers)

country = [pn[:2] for pn in clean_frame['PN']]

clean_frame['country'] = country

date = [pi[-11:] for pi in clean_frame['PI']]

clean_frame['date'] = date

month = [date[-8:] for date in clean_frame['date']]
year = [date[-4:] for date in clean_frame['date']]

clean_frame['month'] = month
clean_frame['year'] = year

clean_frame = clean_frame.drop(columns=['PN', 'PI'])

clean_frame.head()

clean_frame.to_csv('Downloads/web_of_science_data.csv')
