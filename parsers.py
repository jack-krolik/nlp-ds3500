from nltk.sentiment.vader import SentimentIntensityAnalyzer
import json
from collections import Counter
import pandas as pd


def read_lrc(filename):
    """
    Reads a lyrical file
    :param filename (string): name of the lrc file to read
    :return (string): song as string
    """

    # read the file
    with open(filename) as f:
        contents = f.read()

    # get the contents of the file as a list of lyrics
    contents = contents.split("\n")

    song = ''

    # for every lyric in contents add it to the song string
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
    """
    Gets a dataframe of sentiment about the song
    :param string: song
    :return (dataframe): sentiment dataframe of the song
    """

    # create a dataframe for the song
    song_df = pd.DataFrame(columns = ['Line Number', 'Lyric', 'Num Words'])

    # get a list of each lyric line
    lyric_list = string.split('\n')

    # intitalize sentiment analyzer
    sid = SentimentIntensityAnalyzer()

    # add each lyric line, the length, and sentiment to the song_df
    for lyric_idx, lyric in enumerate(lyric_list):

        lyric_df = pd.Series({'Line Number': lyric_idx + 1, 'Lyric': lyric, 'Num Words': len(lyric),
                              'Sentiment': sid.polarity_scores(lyric)})

        song_df = pd.concat([song_df, lyric_df.to_frame().T], ignore_index=True)

    return song_df






