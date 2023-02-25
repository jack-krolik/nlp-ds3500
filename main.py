from NLP2 import NaturalLanguage
import os


def main():
    # test

    nlp_list = []

    for file in os.listdir('Songs'):
        song = os.path.join('Songs', file)
        nlp_list.append(song)

    print(nlp_list)

    nlp = NaturalLanguage(nlp_list, parser='read_lrc')

    print(nlp)

"""
    nlp = NaturalLanguage()

    nlp.load_text('Songs/Dont-stop-me-now-by-Queen.lrc', parser=read_lrc)
    print(nlp.data)
    print(nlp.song)
    print(nlp.df.to_string())

    # nlp.load_text('Songs/6-Foot-7-foot-by-Lil-Wayne.lrc', parser=read_lrc, explicit=False)
    # print(nlp.data)
    # print(nlp.song)
    # print(nlp.df.to_string())
"""

main()