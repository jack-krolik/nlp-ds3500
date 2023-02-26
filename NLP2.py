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
import seaborn as sns
from ErrorHandler import *
import plotly.graph_objects as go



class NaturalLanguage:
    """ Class for Natural Language Processor for Songs"""

    def __init__(self, filenames, labels=None, parser=None, **kwargs):
        # manage data about the different texts that we register with the framework
        self.filenames = filenames
        self.data = defaultdict(lambda: {})
        self.lyrics = []
        self.load_text(filenames, labels, parser, **kwargs)

    @staticmethod
    def clean_string(string, explicit=False, add_words=None):
        """ static method to clean the song
        Args:
            string (str): song string
            explicit (bool): whether to remove explicit words
            add_words (list of str): additional words to be removed from song

        Returns:
            final_string (str): clean song as a string
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
        """ static method to get results of the song

        Args:
            string (str): song as a string

        Returns:
            results (dict): dictionary containing word frequency and total words in song
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
        """ static method to parse a txt file

        Args:
            filename (str): name of file

        Returns:
            song (str): lyrics of the song
            results (dict): stats about the song
        """
        # if the file is not found throw an error otherwise get the song lyrics
        try:
            with open(filename) as f:
                song = f.read()
        except FileNotFoundError:
            raise FileNotFound(filename)

        # get the results
        results = NaturalLanguage._get_results(song, **kwargs)

        return song, results

    def df_parser(self, idx, **kwargs):
        """ gets a dataframe of sentiment about the song

        Args:
            idx (int): index of lyric list

        Returns:
            song_df (dataframe): sentiment dataframe of the song
        """
        cols = kwargs.get("cols", ['Line Number', 'Lyric', 'Num Words'])
        assert isinstance(cols, list), "cols parameter must be a list"

        # create a dataframe for the song
        song_df = pd.DataFrame(columns=cols)

        lyric_list = self.lyrics[idx].split('\n')

        # initialize sentiment analyzer
        sid = SentimentIntensityAnalyzer()

        # add each lyric line, the length, and sentiment to the song_df
        for lyric_idx, lyric in enumerate(lyric_list):
            lyric_df = pd.Series({'Line Number': lyric_idx + 1, 'Lyric': lyric, 'Num Words': len(lyric),
                                  'Sentiment': sid.polarity_scores(lyric)['compound']})

            song_df = pd.concat([song_df, lyric_df.to_frame().T], ignore_index=True)

        return song_df

    def load_text(self, filenames, labels=None, parser=None, **kwargs):
        """ Register a document with the framework

        Args:
            filenames (string): filename of the song
            labels (string): label to assign to the song
            parser (func): parser function to use
            kwargs: other keywords passed by user
        """
        assert isinstance(filenames, list), f"filenames parameter must be a list, got type {type(filenames)}"

        if labels:
            assert len(labels) == len(filenames), f"labels must be the same length as filenames " \
                                                 f"got {len(labels)} labels and {len(filenames)} filenames"
        else:
            labels = filenames

        for idx, filename in enumerate(filenames):

            # do parsing of lrc file
            if parser == 'read_lrc':
                if parser == 'read_lrc':
                    song = read_lrc(filename)

                    results = NaturalLanguage._get_results(song, **kwargs)

                else:
                    assert True, 'Parser Not Found'

            # do default parsing of standard .txt file
            else:

                song, results = NaturalLanguage._default_parser(filename, **kwargs)

            # Save / integrate the song lyrics and song data we extracted from the file
            # into the internal state of the framework
            self.lyrics.append(song)

            for k, v in results.items():
                self.data[labels[idx]][k] = v

            self.data[labels[idx]]['df'] = self.df_parser(idx, **kwargs)

        # get rid of the newline strings in the song lyrics
        for idx, song in enumerate(self.lyrics):
            self.lyrics[idx] = song.replace("\n", " ")

    def wordcount_sankey(self, num_words=10):
        """ creates a sankey diagram of the top ten words used in lyrics

        Args:
            num_words (int): number of words to be shown in sankey
        """

        # double checks if num_words in an integer
        assert type(num_words) == int, f"Make sure 'num_words' is an int ({type(num_words)} given)"

        # creates dict to count word frequency across all songs
        top_ten_dict = defaultdict(lambda: 0)

        # finds word count across all songs
        for value in self.data.values():
            for word, count in dict(value['wordcount']).items():
                top_ten_dict[word] += count

        # sorts words by most used, cutting off words ranked below index num_words
        ds_top = pd.Series(top_ten_dict).sort_values(ascending=False)[:num_words]

        # makes list of most used words
        top = list(ds_top.index)

        # creates link dictionary to be used in sankey creation
        sankey_dict = {'source': [],
                       'target': [],
                       'value': []}

        # fills list into the sankey dictionary
        for key, value in self.data.items():
            for word, count in dict(value['wordcount']).items():
                if word in top:
                    sankey_dict['source'].append(key)
                    sankey_dict['target'].append(word)
                    sankey_dict['value'].append(count)

        # creates a set for all sources and targets, and setting that as label for nodes
        string_set = list(set(list(sankey_dict['source']) + list(sankey_dict['target'])))
        node = {'label': string_set}

        # converts every string into a unique number using index in string set
        for key in ['source', 'target']:
            for i in range(len(sankey_dict[key])):
                sankey_dict[key][i] = string_set.index(sankey_dict[key][i])

        # plots and displays sankey diagram
        sk = go.Sankey(link=sankey_dict, node=node)
        fig = go.Figure(sk)
        fig.show()

    def plot_sentiment(self, songs_list, songs_names):
        """
        Plot every songs' sentiment scores by line in subplots

        :param songs_list (list): list of lrc song file path
        :param songs_names (list): list of song names
        :return: subplots of each song
        """
        # Determine the number of rows and columns needed for the subplots

        if len(songs_list) == 1:
            comp = visual_sentiment(songs[0])
            rolling_avg = pd.Series(comp).rolling(window=4).mean()

            plt.plot(range(len(comp)), comp, c='black', label='total sentiment score')
            plt.plot(range(len(rolling_avg)), rolling_avg, c='red', label='rolling average')
            plt.title(songs_names[0])
            plt.xlabel('Lyric Line Number')
            plt.ylabel('Sentiment Score')
            plt.legend(loc='upper left')

        else:

            plot_subplots()

    def plot_subplots(self):

        n_songs = len(songs_list)
        n_cols = min(n_songs, 3)
        n_rows = (n_songs + n_cols - 1) // n_cols

        # Create the figure and subplots
        sns.set()
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(30, 5 * n_rows))
        fig.subplots_adjust(hspace=0.5, wspace=0.2)

        if n_songs == 2 or n_songs == 3:
            for i, (song, name) in enumerate(zip(songs_list, songs_names)):
                comp = visual_sentiment(songs[i])
                rolling_avg = pd.Series(comp).rolling(window=4).mean()
                axs[i].plot(range(len(comp)), comp, c='black', label='total sentiment score')
                axs[i].plot(range(len(rolling_avg)), rolling_avg, c='red', label='rolling average')
                axs[i].set_title(songs_names[i])
                axs[i].set_xlabel('Lyric Line Number')
                axs[i].set_ylabel('Sentiment Score')
                axs[i].legend(loc='upper left')

        else:
            # Loop over each song and plot its sentiment scores on a subplot
            for i, (song, name) in enumerate(zip(songs_list, songs_names)):
                # calculate row and column index based on song index
                row = i // n_cols
                col = i % n_cols

                # Get sentiment scores and rolling average for the current song
                comp = visual_sentiment(song)
                rolling_avg = pd.Series(comp).rolling(window=4).mean()

                # Plot the sentiment scores and rolling average on the subplot
                axs[row, col].plot(range(len(comp)), comp, c='black', label='total sentiment score')
                axs[row, col].plot(range(len(rolling_avg)), rolling_avg, c='red', label='rolling average')
                axs[row, col].set_title(name)
                axs[row, col].set_xlabel('Lyric Line Number')
                axs[row, col].set_ylabel('Sentiment Score')
                axs[row, col].legend(loc='upper left')

            # Remove any unused subplots
            for i in range(n_songs, n_rows * n_cols):
                row = i // n_cols
                col = i % n_cols
                axs[row, col].set_visible(False)

        plt.show()



# nlp = NaturalLanguage()

# nlp.load_text(['Songs/Dont-stop-me-now-by-Queen.lrc', 'Songs/6-Foot-7-foot-by-Lil-Wayne.lrc'], parser=read_lrc)
# print(nlp.data)
# print(nlp.songs)
# print(nlp.df)
# print(nlp.df['Songs/6-Foot-7-foot-by-Lil-Wayne.lrc'])

# nlp.load_text('Songs/6-Foot-7-foot-by-Lil-Wayne.lrc', parser=read_lrc, explicit=False)
# print(nlp.data)
# print(nlp.song)
# print(nlp.df.to_string())
