from NLP import NaturalLanguage
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

    # song_names = ['joe', 'jack']

    nlp = NaturalLanguage(['Songs/Drake - Hotline Bling.lrc'], parser='read_lrc')
    print(nlp.lyrics)
    # nlp.plot_sentiment(nlp.filenames[:5])

    # nlp = NaturalLanguage(filenames=['Songs/Drake - Hotline Bling.lrc'], parser='read_lrc')
    # print(nlp.lyrics[])
    # print(nlp.data['Songs/Drake - Hotline Bling.lrc'])

    # songs = ['Songs/Dont-stop-me-now-by-Queen.lrc', 'Songs/Katy Perry - Firework.lrc',
    #          'Songs/Thunderstruck by AC-DC.lrc', 'Songs/Elvis Presley - Hound Dog.lrc',
    #          'Songs/Drives license by Olivia rodrigo.lrc', 'Songs/Thunderstruck by AC-DC.lrc',
    #          'Songs/Drake - Hotline Bling.lrc']
    # names = ['idk', 'idc', 'lol', 'stupid', 'abc', 'efg', 'troll']

    # nlp = NaturalLanguage(filenames=[], parser='read_lrc')
    # nlp.plot_sentiment(songs)

    songs = ['Songs/Dont-stop-me-now-by-Queen.lrc', 'Songs/Thunderstruck by AC-DC.lrc',
             'Songs/Numb by Linkin Park.lrc', 'Songs/Katy Perry - Firework.lrc',
             'Songs/Drives license by Olivia rodrigo.lrc',
             'Songs/Bruno Mars, Anderson .Paak, Silk Sonic - Smokin Out The Window [Official Music Video].lrc',
             'Songs/Drake - Hotline Bling.lrc',
             'Songs/J. Cole - p r i d e . i s . t h e . d e v I l (Official Audio)(MP3_160K).lrc',
             'Songs/Hypnotize by The Notorious B.I.G..lrc', 'Songs/Elvis Presley - Hound Dog.lrc',
             'Songs/Mr blue sky by Electric Light Orchestra.lrc', 'Songs/Bob Dylan - Like a Rolling Stone.lrc']

    names = ['Don\'t Stop Me Now (Rock)', 'Thunderstruck (Rock)', 'Numb (Rock)',
             'Firework (Pop)', 'Driver\'s License (Pop)', 'Smokin Out The Window (Pop)',
             'Hotline Bling (Rap)', 'Pride is the Devil (Rap)', 'Hypnotize (Rap)',
             'Hound Dog (Oldies)', 'Mr. Blue Sky (Oldies)', 'Like a Rolling Stone (Oldies)']

    nlp = NaturalLanguage(filenames=songs)
    nlp.plot_repetition(filenames=songs, labels=names)
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

