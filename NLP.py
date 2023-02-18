"""
Ben Ecsedy, Jack Krolik, Teng Li, Joey Scolponeti
DS3500
NLP
"""


from collections import Counter, defaultdict
import nltk
import matplotlib.pyplot as plt
import nltk.corpus
from parsers import read_lrc, df_parser
import string as s



class NaturalLanguage:
    """ Class for Natural Language Processor for Songs"""

    def __init__(self):
        # manage data about the different texts that we register with the framework
        self.data = defaultdict(dict)
        self.song = ""

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
        with open(filename) as f:
            contents = f.read()

        song = contents
        results = NaturalLanguage._get_results(contents, **kwargs)

        return song, results

    def _save_results(self, label, results):
        """ Integrate parsing results into internal state
        label: unique label for a text file that we parsed
        results: the data extracted from the file as a dictionary attribute-->raw data
        """

        for k, v in results.items():
            self.data[k][label] = v

    def load_text(self, filename, label=None, parser=None, **kwargs):
        """
        Register a document with the framework
        :param filename (string): filename of the song
        :param label (string): label to assign to the song
        :param parser (func): parser function to use
        :param kwargs: other keywords passed by user
        """
        if parser is None:  # do default parsing of standard .txt file
            song, results = NaturalLanguage._default_parser(filename, **kwargs)

        elif parser == read_lrc:
            song = read_lrc(filename) # do parsing of lrc file

            results = NaturalLanguage._get_results(song, **kwargs)

        else:
            song = ""
            results = parser(filename)

        if label is None:
            label = filename

        # Save / integrate the song and song data we extracted from the file
        # into the internal state of the framework
        self.song = song
        self._save_results(label, results)


# test
nlp = NaturalLanguage()

# nlp.load_text('Songs/Dont-stop-me-now-by-Queen.lrc', parser=read_lrc)
# print(nlp.data)
# print(nlp.song)
# print(df_parser(nlp.song))


nlp.load_text('Songs/6-Foot-7-foot-by-Lil-Wayne.lrc', parser=read_lrc, explicit=False)
print(nlp.data)
print(nlp.song)
print(df_parser(nlp.song))
