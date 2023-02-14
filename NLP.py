"""
Core framework class for NLP Comparative Analysis
"""

from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from parsers import read_lrc, df_parser
import string as s


class NaturalLanguage:

    def __init__(self):
        # manage data about the different texts that
        # we register with the framework
        self.data = defaultdict(dict)

    @staticmethod
    def clean_string(string):

        string = string.replace('\n', ' ')

        clean_string = "".join([letter.lower() for letter in string if letter not in s.punctuation])

        return clean_string

    @staticmethod
    def _get_results(string):
        """ Integrate parsing results into internal state
        label: unique label for a text file that we parsed
        results: the data extracted from the file as a dictionary attribute-->raw data
        """
        lyrics = NaturalLanguage.clean_string(string)
        song = lyrics.split()
        results = {
            'wordcount': Counter(song),
            'numwords': len(song)
        }
        return results

    @staticmethod
    def _default_parser(filename):
        with open(filename) as f:
            contents = f.read()

        results = NaturalLanguage._get_results(contents)

        return results

    def _save_results(self, label, results):
        """ Integrate parsing results into internal state
        label: unique label for a text file that we parsed
        results: the data extracted from the file as a dictionary attribute-->raw data
        """

        for k, v in results.items():
            self.data[k][label] = v

    def load_text(self, filename, label=None, parser=None):
        """ Register a document with the framework """
        if parser is None:  # do default parsing of standard .txt file
            results = NaturalLanguage._default_parser(filename)

        elif parser == read_lrc:
            f = read_lrc(filename)
            results = NaturalLanguage._get_results(f)

        else:
            results = parser(filename)

        if label is None:
            label = filename

        # Save / integrate the data we extracted from the file
        # into the internal state of the framework

        self._save_results(label, results)

    def compare_num_words(self):
        num_words = self.data['numwords']
        for label, nw in num_words.items():
            plt.bar(label, nw)
        plt.show()



# test
nlp = NaturalLanguage()

nlp.load_text('Songs/Dont-stop-me-now-by-Queen.lrc', parser = read_lrc)
nlp.data