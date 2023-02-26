"""
Ben Ecsedy, Jack Krolik, Teng Li, Joey Scolponeti
DS3500
NLP
"""


from collections import Counter, defaultdict
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import nltk.corpus
from parser import read_lrc
import string as s
import pandas as pd
from ErrorHandler import *





class NaturalLanguage:
    """ Class for Natural Language Processor for Songs"""

    def __init__(self, filename):
        # manage data about the different texts that we register with the framework
        self.data = {}
        self.song = ""
        self.df = None
        self.filename = filename


    @staticmethod
    def clean_string(string, explicit=False, add_words=None):
        """
        static method to clean the song
        :param string: song string
        :param explicit: whether to remove explicit words
        :param add_words: additional words to be removed from song
        :return: clean song as a string
        """

        # get the stopwords
        with open("words/stopwords.txt", "r") as file:
            stopwords = file.read().split("\n")

        # get the explicit words list if set to True and append it to stopwords list
        if not explicit:
            with open("words/badwords.txt", "r") as file:
                badwords = file.read().split("\n")
            stopwords += badwords

        # append add words if any to stopwords
        if add_words is not None:
            assert isinstance(add_words, list), "add_words parameter must be a list"
            stopwords += add_words

        # get the song as a single line string
        string = string.replace('\n', ' ')

        # remove punctuation from the song
        clean_string = "".join([letter.lower() for letter in string if letter not in s.punctuation])

        # remove stopwords from song
        clean_string_without_sw = [word for word in clean_string.split() if word not in stopwords]

        # return the cleaned string as a single line string
        return " ".join(clean_string_without_sw)

    @staticmethod
    def _get_results(string, **kwargs):
        """
        static method to get results of the song
        :param string: song as a string
        :return (dict): result statistics of song
        """

        # get the cleaned song
        cleaned_song = NaturalLanguage.clean_string(string, explicit=kwargs.get("explicit", False),
                                                    add_words=kwargs.get("add_words", None)).split()

        # get the dictionary of results
        results = {
            'wordcount': Counter(cleaned_song),
            'numwords': len(cleaned_song)
        }

        return results

    @staticmethod
    def _default_parser(filename, **kwargs):
        """
        static method to parse a txt file
        :param filename: name of file
        :return (song): the song and the result statistics
        """
        try:
            with open(filename) as f:
                contents = f.read()
        except:
            raise FileNotFound(filename)

        song = contents
        results = NaturalLanguage._get_results(contents, **kwargs)

        return song, results

    def _save_results(self, label, results):
        """ Integrate parsing results into internal state
        label: unique label for a text file that we parsed
        results: the data extracted from the file as a dictionary attribute-->raw data
        """

        for k, v in results.items():
            self.data[k] = v

    def df_parser(self, cols=['Line Number', 'Lyric', 'Num Words']):
        """
        Gets a dataframe of sentiment about the song
        :param string: song
        :param cols (list): column names for song dataframe
        :return (dataframe): sentiment dataframe of the song
        """
        assert isinstance(cols, list), "cols parameter must be a list"
        # create a dataframe for the song
        song_df = pd.DataFrame(columns=cols)

        # get a list of each lyric line
        lyric_list = self.song.split('\n')

        # initialize sentiment analyzer
        sid = SentimentIntensityAnalyzer()

        # add each lyric line, the length, and sentiment to the song_df
        for lyric_idx, lyric in enumerate(lyric_list):
            lyric_df = pd.Series({'Line Number': lyric_idx + 1, 'Lyric': lyric, 'Num Words': len(lyric),
                                  'Sentiment': sid.polarity_scores(lyric)})

            song_df = pd.concat([song_df, lyric_df.to_frame().T], ignore_index=True)

        return song_df

    def load_text(self, label=None, parser=None, **kwargs):
        """
        Register a document with the framework
        :param filename (string): filename of the song
        :param label (string): label to assign to the song
        :param parser (func): parser function to use
        :param kwargs: other keywords passed by user
        """
        if parser is None:  # do default parsing of standard .txt file
            song, results = NaturalLanguage._default_parser(self.filename, **kwargs)

        elif parser == read_lrc:
            song = read_lrc(self.filename) # do parsing of lrc file

            results = NaturalLanguage._get_results(song, **kwargs)

        else:
            song = ""
            results = parser(self.filename)

        if label is None:
            label = self.filename

        # Save / integrate the song and song data we extracted from the file
        # into the internal state of the framework
        self.song = song
        self._save_results(label, results)
        self.df = self.df_parser(cols=kwargs.get("cols", ['Line Number', 'Lyric', 'Num Words']))


# test
nlp = NaturalLanguage('Songs/Dont-stop-me-now-by-Queen.lrc')

nlp.load_text(parser=read_lrc)
print(nlp.data)
print(nlp.song)
print(nlp.df.to_string())


# nlp.load_text('Songs/6-Foot-7-foot-by-Lil-Wayne.lrc', parser=read_lrc, explicit=False)
# print(nlp.data)
# print(nlp.song)
# print(nlp.df.to_string())
