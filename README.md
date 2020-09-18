# lda_topic_modeling

## Files
lda_topic_modeling.py  
lda_topic_modeling.ipynb  

## Description
Both files (the Python file and the Jupyter Notebook) are the same code.  
These files contain a Python class that:  
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


## Input Data
The input data is designed to be the results of a query from a patent database.  

Spefically, one would query the database with specific search terms (related to the technological area of interest).  

The database results would be saved as a .csv or .xlsx.

The data should, at least, consist of the following columns:
- Patent abstract
- Patent source country
- Patent date
The code is currently designed for database results from a specific database so the code would need to be adjusted to take input data from other databases.  

## Required Packages
- pandas
- seaborn
- gensim
- nltk
- matplotlib
- numpy
- datetime
- csv
