import pandas as pd
import seaborn as sns
from gensim.corpora import Dictionary
from gensim.models import TfidfModel, LdaMulticore
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer, LancasterStemmer
import matplotlib.pyplot as plt
from nltk.corpus import wordnet
from nltk import pos_tag
import numpy as np
np.random.seed(2018)
import nltk
from datetime import datetime
import csv

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

stemmer1 = SnowballStemmer('english')
stemmer2 = LancasterStemmer()

'''
code largely appropriated from these websites
visualizations:
https://jeriwieringa.com/2017/06/21/Calculating-and-Visualizing-Topic-Significance-over-Time-Part-1/

topics modeling:
https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24

raising errors:
https://docs.python.org/3/tutorial/errors.html
'''


class Error(Exception):
    '''Base class for exceptions in this module.'''
    pass


class InputError(Error):
    '''Exception raised for errors in the input.
    
    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    '''
    
    def __init__(self, expression, message):
        self.expression = expression
        self.message = message



class TopicModel():
    
    def __init__(self, filename, sort_by = 'year', num_topics=15):
        
        self.sort_by = sort_by
        
        '''
        importing the data
        our import process is different for .csv and .xlsx documents.
        the goal is to end with the data columns we want and the search terms from the patent database.
        the steps to accomplish this differ for .csv and .xlsx documents.
        .csv steps
            -remove first row of export (a descriptor, not a header)
            -extract search terms from that descriptor
        .xlsx steps
            -import first line of .csv document, containing search terms
            -extract search terms from that data
            -import entire .csv file, skipping the first two lines
        '''
        if filename.endswith('.csv'):
            searchterm_df = pd.read_csv(filename, nrows = 1)
            self.searchterms = self.get_searchterms_csv(searchterm_df)
            
            my_data = []
            with open('Downloads\patentpulltestfacialrecognition071719.csv', newline='', encoding='utf8') as f:
                reader = csv.reader(f)
                for row in reader:
                    my_data.append(row)
            self.data = pd.DataFrame(my_data[2:])
            new_header = self.data.iloc[0] # grab the first row for the header
            self.data = self.data[1:] # take the data lwithout the header row
            self.data.columns = new_header # set the header row as the df header
        
        elif filename.endswith('.xlsx'):
            self.data = pd.read_excel(filename, skiprows=1)
            
            searchterm_df = pd.read_excel(filename, nrows=1)
            self.searchterms = self.get_searchterms_xlsx(searchterm_df)
        
        else:
            raise InputError(filename, 'filename should end in .csv or .xlsx')
        
        '''
        after these steps, the steps for both file types are the same
            -we stem our search terms
            -we select our relevant columns: abstracts, source info, and application date.
            -the 'Assignee - Original - Country/Region' is not useful so we pull out country codes from it.
            -depending on the parameters, we trim 'Application Date' down to Year-Month, just Year, or leave it as Year-Month-Day.
        '''
        
        # now let's stem the search terms so they align with our stemmed data
        self.cleaned_terms = []
        for term in self.searchterms:
            self.cleaned_terms.append(self.stem2(term))
        
        # select relevant columns
        # we are interested in Abstract DWPI (our main focus of analysis),
        # Assignee - Original - Country/Region (indicates the source country),
        # and Application Date.
        # We use Abstract DWPI instead of just DWPI because it is generally cleaner and more informative.
        self.documents = self.data[['Abstract - DWPI', 'Assignee - Original - Country/Region', 'Application Date']]
        
        # Assignee - Original - Country/Region consists of company names and country codes.
        # We only want the country codes.
        countries = []
        for i in self.documents['Assignee - Original - Country/Region']:
            if len(str(i).split(',')[1]) > 1: # to make sure our list index is not out of range in the next line
                if len(str(i).split(',')[1]) > 1: # same thing
                    country = str(i).split(',')[1][:2] # the two characters following the comma are most consistently country codes
                    countries.append(country)
                else:
                    country.append('n/a')
            else:
                countries.append('n/a')
        self.documents['Country'] = countries
        self.documents.drop(['Assignee - Original - Country/Region'], axis = 1)
        # attribute of unique country codes for user convenience
        self.unique_countries = self.documents['Country'].unique()
        
        # Application Date's format is YYYY-MM-DD but this may be too granular to see trends in the data.
        # If the user specifies sort_by = 'Month', we only care about YYYY-MM so we cut off the -DD.
        old_height, _ = self.documents.shape
        if self.sort_by == 'month':
            self.documents['Application Date'] = self.documents['Application Date'].apply(lambda x: str(x)[:7])
        # If the user specifies sort_by 'Year', we only care about YYYY so we cut off the -MM-DD.
        elif self.sort_by == 'year':
                self.documents['Application Date'] = self.documents['Application Date'].apply(lambda x: str(x)[:5])
        # If the user specifies sort_by = 'Day', we don't need to change anything.
        elif self.sort_by == 'day':
            pass
        else:
            raise InputError(self.sort_by, 'sort_by must be set as \'year\', \'month\', or \'day\'.')
        # to_datetime converts the date string to a datetime object, telling the dataframe that 2015-04 comes after 2015-03
        self.documents['Application Date'] = pd.to_datetime(self.documents['Application Date'], errors='coerce')
        # to_datetime only accepts dates from around 1600 to 2200. errors='coerce' makes dates outside this range in NaT values
        
        # we remove the bad values here
        self.documents = self.documents.dropna(subset=['Application Date'])
        new_height, _ = self.documents.shape
        # to make sure we don't remove too many rows (which would indicate a larger problem), we print how many rows we removed
        height_change = old_height - new_height
        print('We removed {0} rows due to bad dates'.format(height_change))

        # indexing the documents, creating a simple ID
        length, width = self.documents.shape
        index_column = list(range(length)) # list from 0 to length of dataframe (i.e. number of abstracts)
        self.documents.insert(0, 'index_pos', index_column) # adds this new ID column to the dataframe
        
        # dropping documents with no abstract
        old_height, _ self.documents.shape
        self.documents['Abstract - DWPI'].reaplace('', np.nan, inplace=True)
        self.clean_documents = self.documents.dropna(subset=['Abstract - DWPI'])
        new_height, _ = self.clean_documents.shape
        height_change = old_height - new_height
        print('We removed {0} rows due to missing abstracts'.format(height_change))
        
        # preprocess the data
        processed_docs = self.clean_documents['Abstract - DWPI'].map(self.preprocess)
        # indexing the words in the corpus
        dictionary = Dictionary(processed_docs)
        # removing extremely rare and common words
        dictionary.filter_extremes(no_below=15, no_above=0.4, keep_n=100000)
        # removes words in fewer than 15 documents or more than 40% of the corpus. Only keeps the first n most frequent words.
        
        # create the model
        # converts documents to bag of words (bow). bow consists of token IDs and their frequency counts for every document.
        bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
        # tf-idf is a measure of word importance, just like frequency count could be considered a measure of importance.
        # However, tf-idf assumes that words that appear in a high percentage of documents are less important, but if they appear frequently in one document, they are more important.
        tfidf = TfidfModel(bow_corpus)
        corpus_tfidf = tfidf[bow_corpus]
        
        # our model using Gensim
        self.lda_model = LdaMulticore(corpus_tfidf, num_topics=num_topics, id2word=dictionary, passes=10, workers=2)
        
        # now we start building the dataframe we will visualize
        # build a key of topics and ids
        topic_words = []
        topic_id = []
        # the topics come in a list format, so this for loop breaks down the list into a string
        for idx, topic in self.lda_model.print_topics(-1):
            words = topic.split('""') # splits
            words = words[1::2]
            topic_words.append(words)
            topic_id.append(idx)
        self.topic_labels = pd.DataFrame(list(zip(topic_id, topic_words)), columns = ['topic_id', 'topic_words'])
        # make the list of topic words into a string
        self.topic_labels['topic_words'] = self.topic_labels['topic_words'].apply(lambda x: str(','.join(x)))
        
        # build a dataframe of how the topics appear in each document
        index_pos = []
        topic_id = []
        topic_weight = []
        count = 0
        for topics in self.lda_model.get_document_topics(bow_corpus):
            for topic in topics:
                index_pos.append(count)
                top_id, top_weight = topic
                topic_id.append(top_id)
                topic_weight.append(top_weight)
            count += 1
        topic_dtm = pd.DataFrame(list(zip(index_pos, topic_id, topic_weight)), columns = ['index_pos', 'topic_id', 'topic_weight'])
        # Normalizing the topic weight dataframe
        # Reorient from long to wide
        dtm = topic_dtm.pivot(index='index_pos', columns='topic_id', values='topic_weight').fillna(0)
        # Divide each value in a row by the sum of the row to normalize the values
        dtm = (dtm.T/dtm.sum(axis=1)).T
        # Shift back to a long dataframe
        dt_norm = dtm.stack().reset_index()
        dt_norm.columns = ['index_pos', 'topic_id', 'norm_topic_weight']
        
        # merging dataframes
        topics_expanded = dt_norm.merge(self.topic_labels, on='topic_id')
        df = topics_expanded.merge(self.clean_documents, on='index_pos', how='left')
        
        # isolating the top topics
        topics = list(self.topic_labels['topic_id'])
        scores = []
        for topic_id in topics:
            # for every topic id, we record its scores
            scores.append(np.mean(df[(df['topic_id'] == topic_id)]['norm_topic_weight']))
        topic_scores = pd.DataFrame(list(zip(topics, scores)), columns=['topic_id', 'scores'])
        self.sorted_topic_scores = topic_scores.sort_values(by=['scores'], ascending = False)
        # sorting the data by date
        self.df = df.sort_values(by=['Application Date'])
    
    
    
    def stem1(self, text):
        '''
        Snowball stemmer used in preprocess()
        Weaker stemmer than Lancaster. Used for stemming corpus.
        '''
        return stemmer1.stem(text)
    
    def stem2(self, text):
        '''
        Lancaster stemmer used in preprocess()
        More aggressive stemmer than Snowball, used to insure search terms do not appear in corpus.
        '''
        return stemmer2.stem(text)
    
    def preprocess(self, text):
        '''
        Preprocessing of data for modeling.
        Tokenizes, removes stopwords and short words, removes puncuation, makes everything lowercase, stems the words, and removes words related to original search terms.
        '''
        result = []
        for token in simple_preprocess(text):
            if token not in STOPWORDS and len(token) > 3:
                stemmed2 = self.stem2(token)
                if stemmed2 not in self.cleaned_terms: # we do not care about our search terms
                    stemmed1 = self.stem1(token)
                    result.append(stemmed1)
        return result
    
    def get_searchterms_xlsx(self, data):
        '''
        returns the original search terms from the patent database
        '''
        terms = []
        searchterms = ['none']
        for item in list(data): # The patent database includes search terms in their top line
            if 'Search results' in str(item):
                searchterms = item
                break
        if searchterms == ['none']:
            return searchterms
        words = searchterms.split('""') # the search terms are in quotations
        words = words[1::2] # after separating by delimiters, we want only every otehr piece
        for word in words:
            individ = words.split(' ') # if two terms were searched together, we want to separate them
            terms = terms + individ
        return terms
    
    def get_searchterms_csv(self, data):
        '''
        returns the original search terms from the patent database
        '''
        str_data = str(data)
        words = str_data.split('""') # the search terms are in quotations
        words = words[1::2] # after separating by delimiters, we want only every otehr piece
        terms = []
        for word in words:
            individ = word.split(' ') # if two terms were searched together, we want to separate them
            terms = terms + individ
        return terms
    
    def line_graph(self, start = None, end = None, country = None, top = 5):
        '''
        This method is the main visualization tool. It visualizes the prominence of the top topics over time based on the norm_topic_weight variable.
        For one year, the y-value of a topic represents that topic's average topic weight for that year, including in abstracts in which it has a weight of zero.
        Parameters
            -start: optional start date for visualization
            -end: optional end date for visualization
            -country: optional country (source of patents) to focus on
            -top: number of top topics to be visualized
        '''
        top_topics = list(self.sorted_topic_scores['topic_id'][0:top]) # gets the top prevalent topics
        # specify order
        sorted_order = []
        graph_df = pd.DataFrame()
        for i in top_topics: # we only want to graph the top topics
            sorted_order.append(self.topic_labels['topics words'][i])
            current_df = self.df[(self.df['topic_id'] -- i)]
            graph_df = pd.concat([graph_df, current_df])
        
        # years, if specified
        if start:
            start = datetime(start, 1, 1)
            graph_df = graph_df[(graph_df['Application Date'] >= start)]
        if end:
            end = datetime(end, 1, 1)
            graph_df = graph_df[(graph_df['Application Date'] < end)]
        # country, if specified
        if country:
            graph_df = graph_df[(graph_df['Country'] == country)]
        else:
            country = 'all'
        
        # the following step insures that the x-axis tick labels are legible and show only the specified date
        unique_dates = graph_df['Application Date'].unique()
        x_values = []
        if self.sort_by == 'year':
            for date in unique_dates:
                year = str(date)[:4]
                x_values.append(year)
        elif self.sort_by == 'month':
            for date in unique_dates:
                month = str(date)[:7]
                x_values.append(month)
        else: # self.sort_by == 'day'
            for date in unique_dates:
                day = str(date)[:10]
                x_values.append(day)
        
        # create pointplot
        p = sns.catplot(x='Application Date', y='norm_topic_weight', kind='point', hue_order=sorted_order, hue='topic_words',
                        col=None, col_wrap=None, col_order=sorted_order, height=5, aspect=1.5, data=graph_df, ci=None)
        p.set(xticklabels=x_values)
        for axis in p.axes.flat:
            for item in axis.get_xticklabels():
                item.set_rotation(90) # rotates the x-axis tick labels
        p.fig.subplots_adjust(top=0.9)
        p.fig.suptitle(t='Average Normalized Topic Weights. Search terms: {0}. Country: {1}'.format(self.searchterms, country), fontsize=16)
        return p
    
    def get_abstracts(self, topic_id, start = None, end = None, country = None, top = 5):
        '''
        This method pulls the ten abstracts that are most made up of the chosen topic.
        These abstracts should reflect the topic the best.
        Parameters
            -start: optional start date
            -end: optional end date
            -country: optional country (source of patents)
            -top: number of top topics
        '''
        abstract_df = self.df[(self.df['topic_id'] == topic_id)]
        abstract_df = abstract_df.sort_values(by = ['norm_topic_weight'], ascending = False)
        # years, if specified
        if start:
            start = datetime(start, 1, 1)
            abstract_df = abstract_df[(abstract_df['Application Date'] >= start)]
        if end:
            end = datetime(end, 1, 1)
            abstract_df = abstract_df[(abstract_df['Application Date'] < end)]
        # country, if specified
        if country:
            abstract_df = abstract_df[(abstract_df['Country'] == country)]
        for i in range(top):
            print(abstract_df['Abstract - DWPI'].iloc[i])
        return abstract_df[:top]
    
    def get_topicdata(self, start = None, end = None, country = None):
        '''
        This method returns a dataframe of the same data as is graphed in the line_graph().
        In other words, for every topic ID, the dataframe shows for every year that topic ID's average topic weight.
        Parameters
            -start: optional start date
            -end: optional end date
            -country: optional country (source of patents)
            -top: number of top topics
        '''
        topic_df = self.df
        if country:
            topic_df = topic_df[(topic_df['Country'] == country)]
        dates = topic_df['Application Date'].unique()
        topic_data = pd.DataFrame()
        topic_data['Date'] = dates
        for topic_id in self.topic_labels['topic_id']:
            results = []
            cur_df = topic_df[(topic_df['topic_id'] == topic_id)]
            for date in topic_data['Date']:
                date_df = cur_df[(cur_df['Application Date'] == date)]
                result = date_df['norm_topic_weight'].mean()
                results.append(result)
            topic_data['{0}'.format(topic_id)] = results
        if start:
            start = datetime(start, 1, 1)
            topic_data = topic_data[(topic_data['Date'] >= start)]
        if end:
            end = datetime(end, 1, 1)
            topic_data = topic_data[(topic_data['Date'] < end)]
        return topic_data
    
    def count_graph(self, start = None, end = None, country = None, output = 'graph'):
        '''
        This method returns either a  graph or dataframe of how many patents there were in each year.
        The purpose of this method is to prevent analysts from putting too much weight on outliers, since the prevalence of a topic in one year may be due to the scarty of documents in that year.
        Parameters
            -start: optional start date
            -end: optional end date
            -country: optional country (source of patents)
            -output: 'graph' or 'table'
        '''
        topic_data = self.clean_documents
        if country:
            topic_data = topic_data[(topic_data['Country'] == country)]
        if start:
            start = datetime(start, 1, 1)
            topic_data = topic_data[(topic_data['Application Date'] >= start)]
        if end:
            end = datetime(end, 1, 1)
            topic_data = topic_data[(topic_data['Application Date'] < end)]
        
        dates = topic_data['Application Date'].unique()
        count_df = pd.DataFrame()
        count_df['Date'] = dates
        counts = []
        for date in count_df['Date']:
            current_df = topic_data[(topic_data['Application Date'] == date)]
            count, _ = current_df.shape
            counts.append(count)
        count_df['Counts'] = counts
        
        count_df = count_df.sort_values(by=['Date'])
        
        if output == 'graph':
            # the following step insures that the x-axis tick labels are legible and show only the specified date
            unique_values = count_df['Date']
            x_values = []
            if self.sort_by == 'year':
                for date in unique_dates:
                    year = str(date)[:4]
                    x_values.append(int(year))
            elif self.sort_by == 'month':
                for date in unique_dates:
                    month = str(date)[:7]
                    x_values.append(month)
            else: # self.sort_by == 'day'
                for date in unique_dates:
                    day = str(date)[:10]
                    x_values.append(day)
            ax = sns.lineplot(x='Date', y='Count', data=count_df)
            # ax.set(xticklables=x_values)
            for item in ax.get_ticklabels():
                item.set_rotation(90) # rotates the x-axis tick labels
            ax.set_title(label='Count of Documents by Year. Search terms: {0}. Country: {1}'.format(self.searchterms, country))
            return ax
        elif output == 'table':
            return count_df
        else:
            raise InputError(output, 'output must be \'graph\' or \'table\'.')



my_model = TopicModel('<input_data_set.csv>', sort_by = 'year')

# Attributes

my_model.data.head()

print(my_model.searchterms)

print(my_model.cleaned_terms)

my_model.documents.head()

print(my_model.unique_countries)

my_model.clean_documents.head()

my_model.topic_labels.head()

my_model.sorted_topic_scores.head()

my_model.df.head()


# Methods

my_model.line_graph()

my_model.line_graph(start=2009, country='US')

my_model.line_graph(start=2009, country='CN')

my_model.get_abstracts(topic_id=0)

topic_data = my_model.get_topicdata()
topic_data.head()

my_model.count_graph()

count_table = my_model.count_graph(output='table')
count_table.head()

'''
08/06/19

Creator: Soren Gran

My website: sorengran.com

Please contact me there if you have any questions or concerns about this code.

Good luck!
'''
