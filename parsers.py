import json
from collections import Counter



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









