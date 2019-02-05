import pandas as pd

df = pd.read_csv('Match.csv')
#print(df.head())

team_id = 8634

#df_BAR = df[df['home_team_api_id'] != 8634]
#df_BAR = df[df['country_id'] != 1]

df_BAR = df.loc[(df['home_team_api_id'] == team_id) | (df['away_team_api_id'] == team_id) ]

df_BAR ["team_goal_scored"] =0
df_BAR ["team_goal_conceded"] =0
df_BAR ["team_win"] =0
df_BAR ["team_loss"] =0
df_BAR ["draw"] =0


for index, row in df_BAR.iterrows():
  

  if df_BAR.at[index, 'home_team_api_id'] == team_id:
    team_goal_scored = df_BAR.at[index, "home_team_goal"]
    team_goal_conceded = df_BAR.at[index, "away_team_goal"]
  else:
    team_goal_scored = df_BAR.at[index, "away_team_goal"]
    team_goal_conceded = df_BAR.at[index, "home_team_goal"]

  df_BAR.at[index, 'team_goal_scored'] = team_goal_scored
  df_BAR.at[index, 'team_goal_conceded'] = team_goal_conceded

  goal_diff = team_goal_scored - team_goal_conceded

  if goal_diff > 0:
    df_BAR.at[index, "team_win"] = 1
  elif goal_diff < 0:
    df_BAR.at[index, "team_loss"] = 1
  else:
    df_BAR.at[index, "draw"] = 1


print (df_BAR)

df_BAR.to_csv("barca_games.csv")
