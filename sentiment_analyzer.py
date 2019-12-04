from textblob import TextBlob, Word, Blobber
import preprocessor as p
from scipy import stats

tweets_test_1 = ["GO ASTROS!!", 'THANK GOD WE WON', "great win for houstin", 'We own the world, thank god', "I love the nationals", "thank god we work"]
tweets_test_2 = ["yayy!!", 'i love the astros', "we won", 'we are awesome', "the nationals are the best things ever"]
astrons_won = [tweets_test_1, tweets_test_2]

bad_tweets_test_1 = ["fuck the nationals", 'why are we so bad', "this sucks", 'next time', "I hate our team", "the astros are sucking"]
bad_tweets_test_2 = ["that was the worst thing i've seen", 'why isnt he doing better', "its okay", 'cmon team whats up']
astros_lost = [bad_tweets_test_1, bad_tweets_test_2]


#-------------------------------ANALYZER----------------------------------------

#-----------------------------WON GAMES-----------------------------------------

all_astros_wins_polarities = []
average_win_polarity_astros = 0
for list in astrons_won:
    astros_wins_polarities = []
    for tweet in list:
        print(tweet)
        analysis = TextBlob(tweet)
        astros_wins_polarities.append(analysis.sentiment.polarity)

    #finding the average polarity of all the tweets after each won game
    astros_mean_polarity_wins = 0
    astros_win_num = len(astros_wins_polarities)
    for polarity in astros_wins_polarities:
        astros_mean_polarity_wins += polarity
    astros_mean_polarity_wins = astros_mean_polarity_wins / astros_win_num
    all_astros_wins_polarities.append(astros_mean_polarity_wins)

win_num_atotalw = len(all_astros_wins_polarities)
for game in all_astros_wins_polarities:
    average_win_polarity_astros += game
average_win_polarity_astros = average_win_polarity_astros / win_num_atotalw
print("The average polarity of the Astros games won:", average_win_polarity_astros)



#--------------------------------LOST GAMES------------------------------------

all_astros_losses_polarities = []
average_loss_polarity_astros = 0
for list in astros_lost:
    astros_losses_polarities = []
    for tweet in list:
        print(tweet)
        analysis = TextBlob(tweet)
        astros_losses_polarities.append(analysis.sentiment.polarity)

    #finding the average polarity of all the tweets after each lost game
    astros_mean_polarity_losses = 0
    astros_loss_num = len(astros_losses_polarities)
    for polarity in astros_losses_polarities:
        astros_mean_polarity_losses += polarity
    astros_mean_polarity_losses = astros_mean_polarity_losses / astros_loss_num
    all_astros_losses_polarities.append(astros_mean_polarity_losses)

loss_num_atotall = len(all_astros_losses_polarities)
for game in all_astros_losses_polarities:
    average_loss_polarity_astros += game
average_loss_polarity_astros = average_loss_polarity_astros / loss_num_atotall
print("The average polarity of the Astros games lost:", average_loss_polarity_astros)


"""
#--------------------------------NATIONALS WON GAMES----------------------------

all_nationals_wins_polarities = []
average_win_polarity_nationals = 0
for list in nationals_won:
    nationals_wins_polarities = []
    for tweet in list:
        print(tweet)
        analysis = TextBlob(tweet)
        nationals_wins_polarities.append(analysis.sentiment.polarity)

    #finding the average polarity of all the tweets after each won game
    nationals_mean_polarity_wins = 0
    nationals_wins_num = len(nationals_wins_polarities)
    for polarity in nationals_wins_polarities:
        nationals_mean_polarity_wins += polarity
    nationals_mean_polarity_wins = nationals_mean_polarity_wins / nationals_wins_num
    all_nationals_wins_polarities.append(nationals_mean_polarity_wins)

win_num_ntotalw = len(all_nationals_wins_polarities)
for game in all_nationals_wins_polarities:
    average_win_polarity_nationals += game
average_win_polarity_nationals = average_win_polarity_nationals / win_num_ntotalw
print("The average polarity of the Nationals games won:", average_win_polarity_nationals)

#------------------------------NATIONALS LOST GAMES-----------------------------

average_loss_polarity_nationals = 0
all_nationals_losses_polarities = []
for list in nationals_lost:
    nationals_losses_polarities = []
    for tweet in list:
        print(tweet)
        analysis = TextBlob(tweet)
        nationals_losses_polarities.append(analysis.sentiment.polarity)

    #finding the average polarity of all the tweets after each lost game
    nationals_mean_polarity_losses = 0
    nationals_losses_num = len(nationals_losses_polarities)
    for polarity in nationals_losses_polarities:
        nationals_mean_polarity_losses += polarity
    nationals_mean_polarity_losses = nationals_mean_polarity_losses / nationals_losses_num
    all_nationals_losses_polarities.append(nationals_mean_polarity_losses)

loss_num_ntotall = len(all_nationals_losses_polarities)
for game in all_nationals_losses_polarities:
    average_loss_polarity_nationals += game
average_loss_polarity_nationals = average_loss_polarity_nationals / loss_num_ntotall
print("The average polarity of the Nationals games lost:", average_loss_polarity_nationals)
"""

#-----------------------------------T-Test--------------------------------------

# compare the polarity of the wins and losses of both teams
# this means __ times out of 100 they are similar polarities, thus very unlikely, so we are confident that these polarities have a significant enough difference at a 10% confidence interval

stat, p = stats.ttest_ind(all_astros_wins_polarities, all_astros_losses_polarities)
print("T-Test for Astros")
print('t=%.3f, p=%.3f' % (stat, p))
