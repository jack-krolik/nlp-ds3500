from nltk.sentiment.vader import SentimentIntensityAnalyzer
import json
from collections import Counter
import pandas as pd


def read_lrc(filename):

    with open(filename) as f:
        contents = f.read()

    contents = contents.split("\n")

    song = ''

    for line in contents:

        line = line.split(']')
        lyric = line[1]

        song = song.strip()
        song += '\n' + lyric

    return song.strip()


def json_parser(filename):
    f = open(filename, 'r')
    raw = json.load(f)
    text = raw['text']
    words = text.split(" ")
    wc = Counter(words)
    num = len(words)
    f.close()
    return {'wordcount': wc, 'numwords': num}


def df_parser(string):
    song_df = pd.DataFrame(columns = ['Line Number', 'Lyric', 'Num Words'])

    lyric_list = string.split('\n')

    sid = SentimentIntensityAnalyzer()

    for lyric_idx, lyric in enumerate(lyric_list):

        lyric_df = pd.Series({'Line Number': lyric_idx + 1, 'Lyric': lyric, 'Num Words': len(lyric),
                              'Sentiment': sid.polarity_scores(lyric)})

        song_df = pd.concat([song_df, lyric_df.to_frame().T], ignore_index=True)

    return song_df






