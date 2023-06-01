#!/usr/bin/env python
# coding: utf-8

# # IMPORTS

# In[1]:


from datetime import datetime
import requests
from json import dumps, loads
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# # API CONNECTION, DATE INTERVAL and RAW DATA

# In[4]:


urlGetAllPlayers = 'https://balldontlie.io/api/v1/stats'


def get_all_players_from_api(params):
    try:
        all_data_players_list = []
        all_data_players = requests.get(urlGetAllPlayers, params=params)
        data_feched = loads(all_data_players.text)
        all_data_players_list.extend(data_feched['data'])
        metadata_all_players = data_feched['meta']
        for n in range(metadata_all_players['next_page'], metadata_all_players['total_pages'] + 1):
            params['page'] = n
            all_data_players = requests.get(urlGetAllPlayers, params=params)
            data_feched = loads(all_data_players.text)
            all_data_players_list.extend(data_feched['data'])
            metadata_all_players = data_feched['meta']
            print(f"current_page: {metadata_all_players['current_page']}, next_page: {metadata_all_players['next_page']}, total_pages: {metadata_all_players['total_pages']}")

        return all_data_players_list
    except Exception as err:
        print(str(err))


if __name__ == "__main__":
    start_date = input("Please type the start date in the following format: 'YYYY-MM-DD': ")
    end_date = input("Please type the end date in the following format: 'YYYY-MM-DD': ")
    data_fetched_players = get_all_players_from_api(params={'start_date': start_date, 'end_date': end_date, 'per_page': 100})


# # DATA CLEANING

# In[3]:


df = pd.DataFrame(data_fetched_players)
df.head()


# ### Starting Dataset

# In[4]:


df.info()


# ## PLAYER, GAME and TEAM'S DATA

# In[5]:


df_player = df["player"].apply(pd.Series)
df_player = pd.DataFrame(df_player)
df_player.head()


# In[6]:


df_game = df["game"].apply(pd.Series)
df_game = pd.DataFrame(df_game)
df_game.head()


# In[7]:


df_team = df["team"].apply(pd.Series)
df_team = pd.DataFrame(df_team)
df_team.head()


# ### Preliminary information

# In[8]:


df_game.info()
df_player.info()
df_team.info()


# ### Player Data

# #### Drop Height and Weight columns from the Player DF due to lack of data

# In[9]:


df_player = df_player.drop(['height_feet', 'height_inches', 'weight_pounds', 'id', 'team_id'], axis = 1)


# #### Making a Full Name column

# In[10]:


df_player['full_name'] = df_player['first_name'] + ' ' + df_player['last_name']
df_player = df_player.drop(['first_name', 'last_name'], axis = 1)


# ### Team Data

# #### Drop full_name from The Team DF due to redundancy

# In[11]:


df_team = df_team.drop(['full_name'], axis = 1)


# #### List of Teams by id

# In[12]:


df_team.groupby('id').first()


# In[13]:


df_team = df_team.drop(['id'], axis = 1)


# ### Game Data

# #### Changing teams' id for the abbreviation

# In[14]:


team_dict = {1: 'ATL', 2: 'BOS',3: 'BKN', 4: 'CHA',5: 'CHI', 6: 'CLE',7: 'DAL', 8: 'DEN',9: 'DET',10: 'GSW',11: 'HOU',12: 'IND', 13: 'LAC', 14: 'LAL', 15: 'MEM', 16: 'MIA', 17: 'MIL', 18: 'MIN', 19: 'NOP', 20: 'NYK', 21: 'OKC', 22: 'ORL', 23: 'PHI', 24: 'PHX', 25: 'POR', 26 : 'SAC', 27 : 'SAS', 28 : 'TOR', 29 : 'UTA', 30 : 'WAS'}
df_game['home_team'] =  df_game['home_team_id'].apply(lambda x: team_dict[x])
df_game['visitor_team'] =  df_game['visitor_team_id'].apply(lambda x: team_dict[x])


# In[15]:


df_game = df_game.drop(['home_team_id', 'visitor_team_id', 'period', 'status', 'time', 'id'], axis = 1)


# ### Main DF

# #### Drop dicts and id columns from the main DF

# In[16]:


df = df.drop(['game', 'player', 'team', 'id'], axis = 1)
df.head()


# ### Changing mins from object to float

# In[17]:


df['min'] = df['min'].astype(float)


# ## Join Player, Team and Game DFs

# In[18]:


df = pd.concat([df,df_player,df_team,df_game], axis = 1)


# In[19]:


df.head()


# In[20]:


df.info()


# #### Creating a column to mark which team won the match

# In[21]:


def win_or_loose(x,y):
    if x > y:
        return 'local'
    else:
        return 'visitor'

df['winner'] = df.apply(lambda x: win_or_loose(x['home_team_score'], x['visitor_team_score']), axis=1)
df[['home_team','home_team_score', 'visitor_team', 'visitor_team_score', 'winner']].head()


# #### A Column that determines whether the player is Local or not

# In[22]:


def home_or_not(x,y):
    if x == y:
        return 'local'
    else:
        return 'visitor'

df['is_local'] = df.apply(lambda x: home_or_not(x['home_team'], x['abbreviation']), axis=1)


# In[23]:


df[['home_team', 'visitor_team', 'is_local', 'full_name', 'name']].head()


# #### Changing date's format 

# In[24]:


df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y/%m/%d')
df['date'].head()


# #### Turning pct values into real pct values

# In[25]:


df[['ft_pct','fg3_pct','fg_pct']] = df[['ft_pct','fg3_pct','fg_pct']] * 100
df[['ft_pct','fg3_pct','fg_pct']].head()


# ### Add a column that indicates whether the player won or not

# In[26]:


def win_or_lose(x,y):
    if x == y:
        return 'win'
    else:
        return 'lose'

df['win_or_lose'] = df.apply(lambda x: win_or_lose(x['winner'], x['is_local']), axis=1)


# ### Dropping last columns

# In[27]:


columns_to_drop = ['city','name','division', 'season', 'postseason', 'home_team_score', 'visitor_team_score', 'date', 'home_team','visitor_team', 'winner']
df = df.drop(columns=columns_to_drop)


# In[28]:


df.head()


# In[29]:


df.info()


# ### Counting of wins

# In[30]:


wins_by_player = df[df['win_or_lose'] == 'win']['full_name'].value_counts().reset_index()
wins_by_player.columns = ['full_name', 'win_count']


# In[31]:


wins_by_player


# In[32]:


df = df.merge(wins_by_player, on='full_name', how='left')
df.head()


# ### Counting of times the player was local

# In[33]:


nro_local = df[df['is_local'] == 'local']['full_name'].value_counts().reset_index()
nro_local.columns = ['full_name', 'local_count']


# In[34]:


nro_visitor = df[df['is_local'] == 'visitor']['full_name'].value_counts().reset_index()
nro_visitor.columns = ['full_name', 'visitor_count']


# In[35]:


nro_visitor


# In[36]:


df = df.merge(nro_local, on='full_name').merge(nro_visitor, on='full_name')
df.head()


# ### Dropping the columns is_local & win_or_lose

# In[37]:


df = df.drop(['win_or_lose','is_local'], axis = 1)


# In[38]:


df.head()


# In[39]:


df.info()


# ### Using means to keep one row for each player

# In[40]:


df_1 = df.groupby('full_name').agg({'ast': 'mean', 'blk': 'mean', 'dreb': 'mean', 'fg3_pct': 'mean', 'fg3a': 'mean', 
                                    'fg3m': 'mean', 'fg_pct': 'mean', 'fga': 'mean', 'fgm': 'mean', 'ft_pct': 'mean',
                                   'fta': 'mean', 'ftm': 'mean', 'min': 'mean', 'oreb': 'mean', 'pf': 'mean',
                                    'pts': 'mean', 'reb': 'mean', 'stl': 'mean','turnover': 'mean'}).reset_index()


# In[41]:


df_1.head()


# ### Keep columns that don't need means

# In[42]:


df_2 = df[['full_name', 'position', 'win_count', 'local_count', 'visitor_count']].drop_duplicates()


# In[43]:


df_2.head()


# ### Merge columns to make the final DF

# In[44]:


df_to_work = df_1.merge(df_2, on='full_name', how='left')
df_to_work.head()


# # Insights of descriptive characteristics

# ## Central Tendency

# ### Excluding columns that will not be taken into account (win_count would be useful if we had more data.)

# In[45]:


columns_to_exclude = ['win_count', 'local_count', 'visitor_count']


# In[46]:


df_exclude = df_to_work.drop(columns_to_exclude, axis=1)


# In[47]:


df_exclude.info()


# In[48]:


df_exclude.describe()


# At first we can notice two things.
# 
# 1. Count has the same value for every variable. This indicates that there are not missing values to treat with.
# 2. The minimum value is also the same for every variable. This can occur when there are players who have either not played or   have played very little, resulting in a consistent minimum value of 0 across the variables.
# 
# Now it is time to see the difference when we increase the number of minutes played in avg.

# In[49]:


players_gt_5=df_exclude[df_exclude['min'] >= 5]
players_gt_5.describe()


# There are few things we can notice:
# 1. 152 players have played less than 5 minutes per game in a lapse of time of 24 days.
# 2. Now there are some minimum values greater than 0.
# 3. For most of the variables, the difference between the mean and the median is really similar to the previous data, but with greater numbers due to the time played. More minutes, more possibilities to have better stats.
# 4. Almost every column apparently does not suggest a symmetric behavior because of the gap between the median and the mean.
# 5. The difference between the maximum and percentile 75 is huge. So in case we needed to fill missing values, we would use the median instead of the mean.
# 6. Knowing that there are outliers on the data, they should be treated in case we wanted to interpret the standard deviation.

# ## Data Distribution

# ### Creating two DataFrames: one with only the statistics of players and another one with both the statistics and the position

# In[50]:


stats_and_position_gt_5 = players_gt_5.drop(['full_name'], axis=1)


# In[51]:


stats_gt_5 = players_gt_5.drop(['full_name', 'position'], axis=1)


# ### Making Histograms

# In[52]:


for column in stats_and_position_gt_5.columns:
    plt.hist(stats_and_position_gt_5[column], bins='auto')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()


# These histograms provide good information about our data, for example:
# 1. Most variables exhibit positive skewness, indicating a longer tail on the right side of the distribution. However, variables such as Ft, Fg, and Fg3 percentages do not appear to be skewed, and the Minutes Played variable also shows no skewness.
# 2. The Fg and Fg3 percentages resemble a normal distribution, indicating that the majority of players have shooting percentages around 45% and 30%, respectively. And we cannot avoid the fact that almost 50 players have a Fg3 close to 0. They probably play as center, because centers have a role focused on defense, rebounding, and scoring closer to the basket.
# 3. By far, the majority of players exclusively play as either Forwards or Guards. With almost 80 players of difference, the Center position has the next highest number of players.

# Now, it is time to see those players with 0% Fg3.

# In[53]:


players_gt_5[players_gt_5['fg3_pct'] == 0]


# As expected, most players with 0% Fg3 are centers.

# #### Skewness and Kurtosis

# In[54]:


skewness = stats_gt_5.apply(pd.Series.skew)
print("Skewness:")
print(skewness)


# It would be good to summarize this part.
# - Skewness between -0.5 and 0.5: Ft,Fg, and Fg3 percentages are in this range, this suggests a relatively small departure from symmetry, but it does not imply perfect symmetry. Minutes played and Personal Fouls will not be interpreted because it does not make sense to interpret the symmetry for them.
# - Skewness greater than 1: This section will focus only on Points, Assists, Rebounds, and Blocks per game. Blocks per game has skewness greater than 2; this is the most skewed variable, and it is positive, indicating a strong right-skew. The other variables have skewness greater than 1 but less than 2, indicating they are right-skewed as well, although not to the same extent as Blocks.  

# In[55]:


kurtosis = stats_gt_5.apply(pd.Series.kurtosis)
print("Kurtosis:")
print(kurtosis)


# This part will have 3 sections:
# - Leptokurtic: Blocks has a kurtosis of 9, indicating a very significant peak or excess kurtosis, which means the distribution is heavily concentrated around its mean (0.361076) with a heavy tail. Also it has a high proportion of extreme values or outliers. This can be seen in its histogram, it has a very pronounced peak and it is around the mean then it suddenly starts to drop.
# 
# - Mesokurtic: Assists has a kurtosis of 3.05, which is slightly greater than 3 (the kurtosis of a normal distribution). This indicates it has a distribution that is more peaked with heavier tails, but the departure is not as pronounced as in Blocks. Also, it is good to say that in the assists' histogram it is concentrated on a value that is slightly greater than 1, but it is less than the mean(2.058382).
# 
# - Platykurtic: Minutes has a kurtosis of -1.05, can still have peaked values, although less pronounced and a broader distribution of the data. I previously stated that I wouldn't interpret minutes, but I did it to show how can be interpreted a kurtosis of less than 0.

# ### Finding outliers on players that played 5 or more minutes on average.

# In[59]:


for column in stats_gt_5.columns:
    boxplot = plt.boxplot(stats_gt_5[column])
    outliers = [flier.get_ydata() for flier in boxplot['fliers']]
    if outliers:
        print(f"Outliers for {column}:")
        for outlier in outliers:
            print(outlier)
    else:
        print(f"No outliers for {column}")
    plt.xlabel(column)
    plt.ylabel('Value')
    plt.show()


# Now, it's possible to visualize a boxplot and its outlier values. With this, we can reaffirm previous statements:
# - Blocks do have several outliers, confirming our earlier observation.
# - Assists have outliers but less than Blocks.
# - Minutes per game does not exhibit any outliers.
# - Fg3_pct, which had a histogram suggesting a normal distribution, is further supported by the absence of outliers.

# ## Correlation

# In[60]:


stats_gt_5.corr()


# At first, you can fear about all the data that is shown, but this part will be focused only on points, so don't worry. Points are the most important thing in basketball and with this correlation matrix, it's possible to identify which variables are positively or negatively associated with points. It's time to highlight some aspects:
# - Assists, rebounds, and steals have a moderate positive correlation (0.59-0.66). If you made many assists, it means you had the ball on many occasions, resulting in some FGAs. This logic applies to rebounds and steals as well. In the case of rebounds, it is not as high as assists because you can grab a defensive rebound and pass it to the player who is in charge of dribbling. In the case of steals, in most occasions, the player who steals the ball tries to finish the play, but this does not happen every time. The player can lose the ball, pass it, or miss the shot.
# - Fg, Fg3 and Fta have high positive correlation, it is not a surprise , actually is very logic. The more you shoot the ball, the more points you will have. It's not something new at all.
# - Funny things to notice are the weak positive correlation of blocks and the very strong correlation of turnovers. Blocks have been highlighted on many sections, and it is kind of disappointing  to see it is not very correlated to Points, this may be caused because Blocks are part of the defense, but the offense. The correlation of Turnovers can have an explanation, big stars tend to have more the ball, so the rival will defend those players better causing turnovers for the star.
# 
# Let's plot it.

# In[63]:


target_column = 'pts'

support_columns = ['ast', 'reb', 'stl', 'fg3a', 'fga', 'fta', 'blk', 'turnover']

for column in support_columns:
    plt.scatter(stats_gt_5[column], stats_gt_5[target_column])
    plt.xlabel(column)
    plt.ylabel(target_column)
    plt.title(f'Scatter Plot: {column} vs {target_column}')
    plt.show()


# Now, it is more clear the relation between those variables. Turnovers and FGa are the ones with most clear relation with Points. The less correlation two variables have, the graphic will be less linear. With this view, it's clearer to see the concentration and behavior of some variables.

# In conclusion, this descriptive project about NBA data has provided valuable insights into various aspects of the game. Through the analysis of different variables, we have uncovered interesting patterns and relationships among the players' performance metrics. Here are the key findings:
# 
# - Player Distribution and Playing Time: We observed that a significant number of players, 152 to be precise, have played less than 5 minutes per game in a 24-day period.
# 
# - Distribution Characteristics: Most variables exhibited positive skewness, indicating a longer tail on the right side of the distribution. However, some variables, such as field goal percentages (Fg, Fg3) displayed a more symmetrical distribution.
# 
# - Correlation Analysis: Through the correlation matrix, we identified several interesting relationships. Assists, rebounds, and steals showed a moderate positive correlation with points, suggesting their influence on scoring. Additionally, there was a strong positive correlation among field goals made (Fgm), three-point field goals made (Fg3m), and free throw attempts (Fta), indicating that players who attempt more shots tend to score more points.
# 
# - Position Distribution: The majority of players exclusively played as either forwards or guards, with a significant difference in numbers. The center position had the next highest number of players.
# 
# - Notable Observations: Blocks displayed a relatively weak positive correlation with points, which suggests that shot-blocking alone may not strongly contribute to a player's scoring output. This could be due to the defensive nature of blocks, which primarily prevent the opposing team from scoring rather than directly contributing to a player's own point production. On the other hand, turnovers exhibited a strong positive correlation with points, indicating that a higher number of turnovers may be associated with increased scoring. This could be attributed to the fact that turnovers often result from aggressive offensive plays or risky passes, which can lead to fast-break opportunities and scoring chances for both teams.
# 
# - Outliers: Blocks were found to have several outliers, indicating exceptional performances in shot-blocking. Assists also had many outliers, although in a smaller proportion. However, minutes per game did not exhibit any outliers, highlighting its consistent distribution.
# 
# Overall, this descriptive project has shed light on various aspects of NBA data, providing valuable insights into player performance, relationships between variables, and distribution characteristics. These findings can serve as a foundation for further analysis and exploration of the dataset.

# In[ ]:




