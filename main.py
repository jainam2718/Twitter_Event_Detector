import os

from EventSegmentClusterer import get_events, get_seg_similarity
from TimeWindow import TimeWindow
from TwitterEventDetector import TwitterEventDetector

import numpy as np
import matplotlib.pyplot as plt

from lexrank import LexRank
from lexrank.mappings.stopwords import STOPWORDS
from path import Path

# Parameters
#original_tweet_dir = 'data/original_tweets/' # end with '/'
clean_tweet_dir = 'data/cleaned_tweets/without_retweets/' # end with '/'
subwindow_dir = 'data/cleaned_tweets/without_retweets/2012-10-12/' # each file is a subwindow in this folder
event_output_dir = 'results/2012-10-1/'
wiki_titles_file = 'data/enwiki-titles-unstemmed.txt'
seg_prob_file = 'data/seg_prob_2012_Oct_11-22.json'
wiki_Qs_file = 'data/WikiQsEng_non_zero_processed.json'

remove_retweets = True
max_segment_length = 4
hashtag_wt = 3
entities_only = False # False --> use #tag and @name only for event detection
default_seg_prob = 0.0000001 # for unknown segments
use_retweet_count = True
use_followers_count = True
n_neighbors = 4
threshold = 4 # for news_worthiness

ted = TwitterEventDetector(wiki_titles_file, seg_prob_file, wiki_Qs_file, remove_retweets, max_segment_length,
                           hashtag_wt, use_retweet_count, use_followers_count, default_seg_prob, entities_only)

# Tweet Cleaning
#ted.clean_tweets_in_directory(original_tweet_dir, clean_tweet_dir)

# Segment tweets and create TimeWindow
print('\nReading SubWindows')
subwindow_files = [f.name for f in os.scandir(subwindow_dir) if f.is_file()]
x=[]
y=[]
xx = 0
I = 0
subwindows = []
for subwindow_name in subwindow_files[:24]: # read timewindow consisting 6 subwindows of 1 hour each
    print('SubWindow:',subwindow_name)
    sw = ted.read_subwindow(subwindow_dir + subwindow_name)
    subwindows.append(sw)
    '''
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    y = [100152, 141187, 133855, 96271, 62978, 57481, 48055, 40249, 32034, 33049, 40838, 50460, 57668, 67140, 75067, 81609, 83307, 79978, 77340, 83366, 87180, 82181, 78335, 76076]
    fig = plt.figure(figsize = (10, 5))
    plt.bar(x, y, color ='maroon',
            width = 0.4)
    
    plt.xlabel("Each hour")
    plt.ylabel("Total number of Tweets")
    plt.title("Tweet per each hour on 12th October")
    plt.show()
    '''
print('Done\n')    

tw = TimeWindow(subwindows)
print(tw)

#next_subwindow = ted.read_subwindow(subwindow_dir + subwindow_files[7])
#tw.advance_window(next_subwindow)
#print(tw)

# Bursty Segment Extraction
print()
bursty_segment_weights, segment_newsworthiness, currentLength = ted.bse.get_bursty_segments(tw)
'''y.append(currentLength-xx)
xx = currentLength
x.append(I)
I += 1
fig = plt.figure(figsize = (10, 5))
plt.bar(x, y, color ='blue',
        width = 0.4)

plt.xlabel("Each hour")
plt.ylabel("Total number of Segments")
plt.title("Tweet Segments per each hour on 12th October")
plt.show()'''
seg_sim = get_seg_similarity(bursty_segment_weights, tw)
# f = open("seg_sim.txt", 'w')
# f.write(str(seg_sim))

# Clustering Bursty Segments
events = get_events(bursty_segment_weights, segment_newsworthiness, seg_sim, n_neighbors)


# dump event clusters along with tweets[cleaned ones :-( ] associated with the segments in the cluster 
print('\nEvents will be saved in', event_output_dir)
if not os.path.exists(event_output_dir):
    os.makedirs(event_output_dir)
event_no = 0
text_full = ""

documents = []
documents_dir = Path('./data/bbc_politics/')

for file_path in documents_dir.files('*.txt'):
    with file_path.open(mode='rt', encoding='utf-8') as fp:
        documents.append(fp.readlines())
        
lxr = LexRank(documents, stopwords=STOPWORDS['en'])

for e, event_worthiness in events:
    text_full = ""
    event_no += 1
    print('\nEVENT:', event_no, 'News Worthiness:', event_worthiness) 
    f = open(event_output_dir + str(event_no) + '.txt', 'w')
    f.write(str(e)+' '+str(event_worthiness)+'\n\n')
    for seg_name in e:
        print(seg_name) 
        f.write('SEGMENT:' + seg_name+'\n')
        for text in set(tw.get_tweets_containing_segment(seg_name)):
            f.write(text+'\n')
            text_full += text
            text_full += "\n"
        f.write('-----------------------------------------------------------\n') 
    f.close()
    

    f = open(event_output_dir+str(event_no)+"_Full.txt", 'w')
    f.write(text_full)
    f.close()
    summary_cont = lxr.get_summary(e, threshold=None)
    print(summary_cont)