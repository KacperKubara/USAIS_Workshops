# Code developed and maintained by Manny Dinssa
#!/usr/bin/env python
# coding: utf-8

# Sentiment Analysis: --- University of Southampton Artifical Intelligence Society version. ---

import pandas as pd     # To handle data
import numpy as np      # For number computing
from pattern.en  import sentiment     # To calculate sentiment

from flask import Flask, jsonify, render_template, request
from flask_restful import reqparse

from bs4 import BeautifulSoup
import requests
import re

app = Flask(__name__)

@app.route('/sa', methods=['POST'])
def postSingle():  
    parser = reqparse.RequestParser()
    parser.add_argument('input')
    args = parser.parse_args()
    
    entry = {
            'input': args['input']
            }
    
    if (entry['input'] != ''):
        return jsonify({'meta': "Single Sentiment Analysis", 'data': {'sentiment': analise_basic_sentiment(entry['input'])}}), 201
    else:
        return "404: No input found", 404

@app.route('/sa/percentages', methods=['POST'])
def postMulti():
            
    parser = reqparse.RequestParser()
    parser.add_argument('input', action = 'append')
    args = parser.parse_args()
    
    entry = {
            'input' : args['input']            
            }
        
    # Pandas dataframe to store entries
    data = pd.DataFrame(data=[inp for inp in entry['input']], columns=['Entry'])
    
    # Calculates the percentages as an array: 0: Positive, 1: Neutral, 2: Negative
    percentages = multiAnalysis(data)

    entry['positive'] = percentages[0]
    entry['neutral'] = percentages[1]
    entry['negative'] = percentages[2]
    
    return jsonify({'meta': 'Percentages Sentiment Analysis',
                    'data': {'percentages': 
                                {'positive': entry['positive'], 
                                 'neutral': entry['neutral'], 
                                 'negative': entry['negative'] }}}), 201
            

# --- Suborutines ---
    
def instagramCaption(url):
	'''
	Scrapping Instagram Captions from public posts
	'''   
    response = requests.get(url)

    soup = BeautifulSoup(response.text, 'html.parser')

    scripts = soup.find_all("script")

    for i in range(len(scripts)):
        if "edge_media_to_caption" in scripts[i].text:
            keyscript = scripts[i].text
        #print (keyscript)
        
        #START: "edge_media_to_caption":{"edges":[{"node":{"text":"
        #END:  "}}]},"caption_is_edited"
        
    try:
        found = re.search('{"node":{"text":"(.+?)"}}]},"caption_is_edited"',keyscript).group(1)
    except AttributeError:
        found = ''    
        
    return found.replace('\\u', ' \\ ')
    
      
def multiAnalysis(data):
    '''
    Each entry is classified as Positive, Negative or Neutral
    '''
    data['Sentiment'] = np.array([ analise_sentiment(entry) for entry in data['Entry'] ])
    
    return percentage_analysis(data)

def analise_basic_sentiment(entry):
    '''
    Basic positive, negative or neutral statement.
    '''
    # Sentiment is given as a value between -1.0 (most negative) and +1.0 (most positive)
    analysis = analise_sentiment(entry)
    
    if analysis > 0:
        return "Positive"
    elif analysis == 0:
        return "Neutral"
    else:
        return "Negative"
    
def analise_sentiment(entry):
    '''
    Polarity between -1.0 and 1.0
    '''    
    # Only interested in polarity of the entry .i.e. first column of sentiment function
    return sentiment(entry)[0]    
    
def percentage_analysis(data):
    '''
    Calculates the percentage of positive, neutral and negative sentiment entries
    '''
    # Calculates percentage by { (first each instance of a given sentiment DIVIDED by total number of entries in data) * 100 }
    positive_percentage = len([ entry for index, entry in enumerate(data['Entry']) if data['Sentiment'][index] > 0 ])*100/len(data['Entry'])
    neutral_percentage = len([ entry for index, entry in enumerate(data['Entry']) if data['Sentiment'][index] == 0 ])*100/len(data['Entry'])
    negative_percentage = len([ entry for index, entry in enumerate(data['Entry']) if data['Sentiment'][index] < 0 ])*100/len(data['Entry'])
    
    # Returns the 3 values within a single array
    return [positive_percentage, neutral_percentage, negative_percentage]
    
if __name__ == '__main__':
    app.run(debug=True, port=8080 )