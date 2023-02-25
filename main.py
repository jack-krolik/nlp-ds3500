from NLP2 import NaturalLanguage
import os


def main():
    # test

    nlp_list = []

    for file in os.listdir('Songs'):
        song = os.path.join('Songs', file)
        nlp_list.append(song)

    song_names = ['Drivers license - Olivia Rodrigo (Pop)', 'Smokin Out The Window - Silk Sonic (Pop)',
              'Firework - Katy Perry (Pop)', 'Dont Stop Me Now - Queen (Rock)', 'Numb - Lincoln Park (Rock)',
              'Thunderstruck - AC/DC (Rock)', 'Hypnotize - Biggie Smalls (Rap)', 'Hotline Bling - Drake (Rap)',
              'Pride is the Devil - J. Cole (Rap)', 'Mr blue sky -  Electric Light Orchestra (Oldies)',
              'Like a Rolling Stone - Bob Dylan (Oldies)', 'Hound Dog - Elvis (Oldies)']

    nlp = NaturalLanguage(nlp_list, parser='read_lrc')

    nlp.sentiment(nlp.filenames[:5])

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