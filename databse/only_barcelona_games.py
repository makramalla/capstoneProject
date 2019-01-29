import pandas as pd

df = pd.read_csv('Match.csv')
#print(df.head())

#df_BAR = df[df['home_team_api_id'] != 8634]
#df_BAR = df[df['country_id'] != 1]

df_BAR = df.loc[(df['home_team_api_id'] == 8634) | (df['away_team_api_id'] == 8634) ]
#df_BAR = df.loc[(df['home_team_api_id'] == 8634)
df_BAR ["home_win"] =0
df_BAR ["away_win"] =0
df_BAR ["draw"] =0


for index, row in df_BAR.iterrows():
  goal_diff = df_BAR.at[index, "home_team_goal"] - df_BAR.at[index, "away_team_goal"]
  
  if goal_diff > 0:
    df_BAR.at[index, "home_win"] = 1
  elif goal_diff ==0:
    df_BAR.at[index, "draw"] = 1
  else:
    df_BAR.at[index, "away_win"] = 1

print (df_BAR)

df_BAR.to_csv("barca_games.csv")
