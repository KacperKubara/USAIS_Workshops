# -*- coding: utf-8 -*-

"""
Pattern.web can be used to extract information from Wikipedia, Facebook, Twitter, etc
Find out more: https://www.clips.uantwerpen.be/pages/pattern-web
"""

from pattern.web import Twitter
from pattern.en import sentiment

twitter = Twitter()
someTweet = None
feeling = ""

for j in range(1):
    for tweet in twitter.search('Teressa May', start=someTweet, count=1):
        print("Tweet: ",tweet.text)
        someTweet = tweet.id
        
        '''So is it Positve, Negative or Neutral? The real questions'''
        
        if sentiment(tweet.text)[0] > 0:
            feeling = "Positive"
        elif sentiment(tweet.text)[0] < 0:
            feeling = "Negative"
        else:
            feeling = "Neutral"
            
        print("Sentiment is: ", feeling, "[", sentiment(tweet.text)[0], "]")
        

''' ---------------------- EXTENSIONS ----------------------

    1) Can you make this work for a Facebook post?
    
    2) Can you make it work for all languages?
    
        Hint: in pattern.en the "en" stands for english
        Hint: Have a look at the google translate library
    
    --------------------------------------------------------  
''' 

