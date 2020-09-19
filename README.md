# lda_topic_modeling

## Files
### python folder
lda_topic_modeling.py  
web_of_science_data.py  

### jupyter_notebook folder
lda_topic_modeling.ipynb  
web_of_science_data.ipynb  

web_of_science_data.csv

## Description
Both folders (the python folder and the jupyter_notebook) contain the same code.  
The lda_topic_modeling files contain a Python class that:  
- imports text data
- preprocesses the data
- runs a topic modeling model on the data using Latent Dirichlet Allocation
- based on the topic modeling, finds trends in the topic data
- returns a line graph of the topic trends over time
- returns a table of the topic trends over time

This code was originally designed to work with patent abstract data.  

The goal was to use topic modeling to identify different topics in patent abstracts, which would represent the technologies or ideas in those abstracts.  

Once the topics are created, the code can track these topics for different countries over the years (or months).  

By analyzing the prevalence of different topics over the years for one country, the code demonstrates technological trends for that country over that time period.  

This can also be done for all patents (regardless of country or time frame).  

The web_of_science_data files contain code that concatenates data from seven .txt tab-delimited files.  

Web of Science only allowed me to export 500 patents at a time. There were 3,475 for my particular search terms ('facial recognition') so I exported seven total files of patent data. This file quickly and easily combines the seven data files into one file. It also isolates and renames the columns I am interested in.

For my purposes, I am only interested in patent abstracts, countries of origin, and application dates.

web_of_science_data.csv contains the data I pulled from Web of Science. I originally pulled this data in seven separate files, and then I combined them using web_of_science_data.ipynb. This data file consists of four columns: 'abstract', 'country', 'date', 'month', and 'year'. 'date', 'month', and 'year' are repetitive but make it easier to use different date formats in the main code file.

## Input Data
The input data is designed to be the results of a query from a patent database.  

Spefically, one would query the database with specific search terms (related to the technological area of interest).  

The lda_topic_modeling code takes .csv files with columns named 'abstract', 'country', 'date', 'month', and 'year' as input.

The data should, at least, consist of the following columns:
- Patent abstract: 'abstract' should consist of strings of patent abstracts.
- Patent source country: 'country', ideally, is a two-letter country code.
- Patent date:
  - 'date' consists of day-month-year dates.
  - 'month' consists of month-year dates.
  - 'year' consists of year dates.

The code is currently designed for database results from Web of Science, which I access through the Claremont Colleges Library. The code may need to be adjusted to take input data from other databases.  

## Required Packages
- pandas
- seaborn
- gensim
- nltk
- matplotlib
- numpy
- datetime
- csv
- glob
