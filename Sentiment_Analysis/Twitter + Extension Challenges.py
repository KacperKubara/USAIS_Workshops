# -*- coding: utf-8 -*-

"""
Pattern.web can be used to extract information from Wikipedia, Facebook, Twitter, etc
Find out more: https://www.clips.uantwerpen.be/pages/pattern-web
"""

from pattern.web import Facebook
from pattern.en import sentiment
from pattern.web import Facebook, NEWS, COMMENTS, LIKES

fb = Facebook(license='568752540312810')
me = fb.profile(id=568752540312810) # user info dict

for post in fb.search(me['id'], type=NEWS, count=100):
    print repr(post.id)
    print repr(post.text)
    print repr(post.url)
    if post.comments > 0:
        print '%i comments' % post.comments 
        print [(r.text, r.author) for r in fb.search(post.id, type=COMMENTS)]
    if post.likes > 0:
        print '%i likes' % post.likes 
        print [r.author for r in fb.search(post.id, type=LIKES)]
"""
facebook = Facebook()
someTweet = None
feeling = ""

for j in range(1):
    for post in facebook.search('Teressa May', start=someTweet, count=1):
        print("Tweet: ",post.text)
        someTweet = post.id
        
        '''So is it Positve, Negative or Neutral? The real questions'''
        
        if sentiment(post.text)[0] > 0:
            feeling = "Positive"
        elif sentiment(post.text)[0] < 0:
            feeling = "Negative"
        else:
            feeling = "Neutral"
            
        print("Sentiment is: ", feeling, "[", sentiment(post.text)[0], "]")
        
"""
''' ---------------------- EXTENSIONS ----------------------

    1) Can you make this work for a Facebook post?
    
    2) Can you make it work for all languages?
    
        Hint: in pattern.en the "en" stands for english
        Hint: Have a look at the google translate library
    
    --------------------------------------------------------  
''' 

