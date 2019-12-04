import textblob
import tweepy
#import preprocessor as p
import preprocessor as p
from preprocessor import clean, tokenize, parse
import datetime


consumer_key = 'MC6IhlxZY5qys2V8iQKGQHPyM'
consumer_key_secret = 'exTsFcPRU8mWNemMoLi9vQNkgeEb9SmESwaAvGxg3w5ssTUnyR'
access_token = '1186688342424879109-efDNkOCRME6DHgd3Egy3QVo4ANVXbd'
access_token_secret = 'JWTAeql3lmSW5u8zHfusdy6Ak7pCaMrpr6Q6aivgod1eM'

#consumer_key = 'fUWgNWiS9rcLDSVvgbTg0Henp'
#consumer_key_secret = 'IoDztDHsm4EvAS8ywLrzPdIwJodwaZBxmlRWyV8lMRb5IZ1R9h'
#access_token = '488802997-UhByVraduyvPTmu7yAjSGh3c7nLPQ2awqrlDR9LA'
#access_token_secret = '0zz55yttzzngMAkQaYyw4vZe5FGyFW9rzt8KybWtC8ltY'

#startPreGame1 = datetime.datetime(2019, 10, 22, 15, 0, 0)
#endPreGame1 =   datetime.datetime(2019, 10, 22, 20, 0, 0)

#startPostGame1 = datetime.datetime(2019, 10, 22, 11, 43, 0)
#endPostGame1 =   datetime.datetime(2019, 10, 23, 4, 43, 0)

#startPreGame2 = datetime.datetime(2019, 10, 23, 15, 0, 0)
#endPreGame2 =   datetime.datetime(2019, 10, 23, 20, 0, 0)

#startPostGame2 = datetime.datetime(2019, 10, 24, 0, 0, 0)
#endPostGame2 =   datetime.datetime(2019, 10, 24, 5, 0, 0)

#startPreGame3 = datetime.datetime(2019, 10, 25, 15, 0, 0)
#endPreGame3 =   datetime.datetime(2019, 10, 25, 20, 0, 0)

#startPostGame3 = datetime.datetime(2019, 10, 26, 0, 0, 0)
#endPostGame3 =   datetime.datetime(2019, 10, 26, 5, 0, 0)

#startPreGame4 = datetime.datetime(2019, 10, 26, 15, 0, 0)
#endPreGame4 =   datetime.datetime(2019, 10, 27, 20, 0, 0)

#startPostGame4 = datetime.datetime(2019, 10, 26, 11, 48, 0)
#endPostGame4 =   datetime.datetime(2019, 10, 27, 4, 48, 0)

#startPreGame5 = datetime.datetime(2019, 10, 27, 15, 0, 0)
#endPreGame5 =   datetime.datetime(2019, 10, 27, 20, 0, 0)

#startPostGame5 = datetime.datetime(2019, 10, 27, 11, 19, 0)
#endPostGame5 =   datetime.datetime(2019, 10, 28, 4, 19, 0)

#startPreGame6 = datetime.datetime(2019, 10, 29, 15, 0, 0)
#endPreGame6 =   datetime.datetime(2019, 10, 29, 20, 0, 0)

#startPostGame6 = datetime.datetime(2019, 10, 29, 11, 37, 0)
#endPostGame6 =   datetime.datetime(2019, 10, 30, 4, 37, 0)

#startPreGame7 = datetime.datetime(2019, 10, 30, 15, 0, 0)
#endPreGame7 =   datetime.datetime(2019, 10, 30, 20, 0, 0)

#startPostGame7 = datetime.datetime(2019, 10, 30, 11, 42, 0)
#endPostGame7 =   datetime.datetime(2019, 10, 31, 4, 42, 0)

#October 22, 2019
#October 30, 2019


auth = tweepy.OAuthHandler(consumer_key, consumer_key_secret)

auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)
#public_nationals_tweets = api.search('Nationals')

pats_tweets = []
for tweet in tweepy.Cursor(api.search, q="Patriots", lang="en").items():
    pats_tweets.append(tweet.text)

print(pats_tweets[0])




#p.set_options(p.OPT.URL, p.OPT.EMOJI)
#clean_nationals_tweets = p.clean('')
#print(public_nationals_tweets.text[0])
#public_nationals_tweets
