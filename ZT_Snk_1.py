# we need a function to load in data, create a new df and format it how we want
# then a function to run the model + store player ranking/rating results (weekly?)

import pickle
import pandas as pd
import numpy as np
from itertools import zip_longest
from scipy.optimize import minimize


'''
Model Workflow:

Matches DataFrame has a logistic function column, where we evaluate logistic_fn(logistic strength p1 - logistic strength p2)
aka the player score differential. The Logistic Strength is what we feed into the model to use as the ratings!

Vary Logistic Strength
Re-calculate player strength differentials
Feed this into logistic function for every game, and update logistic function column of matches df
Update result functions column of matches df
calculate the log likelihood
-> repeat until LL is maximised
'''

matches_data = pd.read_csv('/Users/zactiller/Documents/Sports Trading/Sports Trading Projects/Snooker/snooker/Database/matches.csv', index_col = 0)
players_data = pd.read_csv('/Users/zactiller/Documents/Sports Trading/Sports Trading Projects/Snooker/snooker/Database/players.csv', index_col = 0)


# TODO: have a file for data cleaning & manipulation
# Then have a file for model functions (creating model, updating model)
# Then have a file for making predictions from the model

def update_player_rankings(df):

    # if df does not have a rank column... raise error -> may have passed incorrect df

    df['Rank'] = df['Logistic Strength'].rank(method='min')
    return df


def copy_and_expand_df(df_type, df):
    if df_type == 'Players Data':
        players = True
        players_strength_df = df.copy(deep=True)

        # Adding new columns to the df
        players_strength_df['Model 1 Rating'] = float('NaN')
        players_strength_df['Logistic Strength'] = 1.000 #logistic strength column - this is what our model tunes to optimise ML
        players_strength_df['Rank'] = players_strength_df['Logistic Strength'].rank(method='min')

        players_strength_df.set_index('Player ID', inplace=True) #set the index of the players data to the player IDs
    else: #if 'Matches Data':
        players = False
        matches_df = df.copy(deep=True)

        # Adding new columns to the df
        matches_df['P1 MOV'] = matches_df['Player 1 Frames'] - matches_df['Player 2 Frames']
        matches_df['Logistic Function'] = float('NaN')

        matches_df.loc[matches_df['P1 MOV'] > 0, 'Match Result'] = 1
        matches_df.loc[matches_df['P1 MOV'] <= 0, 'Match Result'] = 0 #TODO: Think about draws

        #TODO: Potentially make the match-result column a free standing function?
        matches_df.loc[matches_df['Match Result'] == 1, 'Result Function'] = matches_df['Logistic Function']
        matches_df.loc[matches_df['Match Result'] == 0, 'Result Function'] = 1-matches_df['Logistic Function']

    return players_strength_df if players else matches_df


def key_fn(x):

    sports_object.player_data['Logistic Strength'] = x

    matches_df = compute_player_rank_differential(sports_object.match_data, sports_object.player_data)  # Creates a col w the diff between player logistic strength values
    matches_df = update_logistic_function(matches_df)  # Applies the logistic function to this strength differential

    sports_object.update_sport_object_data_attributes(matches_df, sports_object.player_data)
    match_LL = -1*sports_object.calc_matches_prob_product(sports_object.match_data)
    print('LL: {}'.format(match_LL))

    return match_LL


def extract_fixture_pID_arr(df):
    """
    :param df: a DataFrame we wish to extract the fixtures (matches) from; in our case, should be the matches DataFrame
    :return: player_id_arr: a numpy array of the fixtures where they are represented in terms of player IDs [ [pID1, pID2],
                                                                                                             [pID3, pID4]...
                                                                                                             [pIDx, pIDy] ],
                            so that later we can use them as handles to grab the player strength from the player dataframe
    """

    player_id_arr = df[['Player 1 ID', 'Player 2 ID']].to_numpy()
    return player_id_arr


def drop_amateurs_collect_pro_pIDs(df):
    """
    :param df: a DataFrame we want to remove amateur players from - in this case, would be the players df
    :return: df: the cleaned DataFrame
            pro_pIDs: a list containing all the pro players IDs
    """

    df = df[df["Turned Pro"] != 0]
    pro_pIDs = list(df.index)

    return df, pro_pIDs


def check_and_remove_inconsistent_pIDs(df, proIDs): #pass the matches dataframe into here to remove matches which are irrelevant
    """
    :param df: a DataFrame (in this case matches) we wish to remove games which do not contain pro players in our list of pros
    :param proIDs: a list of pro players

    :return: df: the cleaned DataFrame
    """
    df = df[df["Player 1 ID"].isin(proIDs)]
    df = df[df["Player 2 ID"].isin(proIDs)]

    return df


def compute_player_rank_differential(matches_df, players_df):
    """
    :param matches_df: matches DataFrame
    :param players_df: players DataFrame

    :return: matches DataFrame, with the player score differential (difference in ranking (Logistic Strength) ) updated
    """
    player_id_arr = extract_fixture_pID_arr(matches_df)

    # Update the score differential (difference in ranking column) of matches DataFrame
    matches_df['Player Score Differential'] = np.array(players_df.loc[player_id_arr[:, 0]]['Logistic Strength']) - \
                                              np.array(players_df.loc[player_id_arr[:, 1]]['Logistic Strength'])

    return matches_df


# TODO: make a v-lookup type function, like above


def update_logistic_function(matches_df):
    # Apply the logistic function to the Player Score Differential for a given game, and re-compute the result function column
    # Note - the Player score differential has been calculated in a 'Vlookup' fashion

    matches_df['Logistic Function'] = matches_df['Player Score Differential'].apply(lambda x: 1 / (1 + np.exp(-x)))

    matches_df.loc[matches_df['Match Result'] == 1, 'Result Function'] = matches_df['Logistic Function']
    matches_df.loc[matches_df['Match Result'] == 0, 'Result Function'] = 1 - matches_df['Logistic Function']

    return matches_df


class MLOptimiser:
    def __init__(self, match_df, player_df):
        self.prob_product = float('NaN')
        self.log_likelihood = float('NaN')

        self.match_data = match_df
        self.player_data = player_df

    def update_sport_object_data_attributes(self, match_df, player_df):
        'Updates the data attributes of the sport object, to be called after we perform operations on the dataframes'

        self.match_data = match_df
        self.player_data = player_df

    def calc_matches_prob_product(self, match_df):
        # To avoid small number errors, take the log of each result function value, then sum up
        self.log_likelihood = np.sum(np.log(np.array(match_df['Result Function'])))
        return self.log_likelihood


    def LL_maximiser(self):

        # start = np.random.randn(len(self.player_data),1)
        start = [1.000]*len(self.player_data)
        result = minimize(key_fn, start, method='Nelder-Mead', options={'disp':True, 'adaptive':True, 'maxiter':100000,
                                                                        'maxfev':100000})

        return result


# BEFORE we make our sports object, apply data cleaning & transformations

players_strength_df = copy_and_expand_df('Players Data', players_data)
matches_df = copy_and_expand_df('Matches Data', matches_data)
players_strength_df, pro_IDs = drop_amateurs_collect_pro_pIDs(players_strength_df)
matches_df = check_and_remove_inconsistent_pIDs(matches_df, pro_IDs)

sports_object = MLOptimiser(matches_df, players_strength_df)
res = sports_object.LL_maximiser()
players_strength_df = update_player_rankings(sports_object.player_data)

pickle.dump(players_strength_df, open('PlayersStrengthDataFrame', 'wb'))

print(res.success)
print(res.message)


