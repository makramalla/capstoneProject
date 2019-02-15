import pandas as pd

df = pd.read_csv('Match.csv')

#Chosen team id = Barcelona
team_id = 8634

#filter the team's games within the dataset
df_team = df.loc[(df['home_team_api_id'] == team_id) | (df['away_team_api_id'] == team_id) ]

#Rename Home and away players to team and opposition team player wihtout changeg the players 
for num in range(1, 12):

  home_player = "home_player_" + str(num) 
  team_player = "team_player_" + str(num)
  away_player = "away_player_" + str(num)
  opp_player = "opp_player_" + str(num)

  df_team.rename(index=str, columns={home_player:team_player}, inplace=True)
  df_team.rename(index=str, columns={away_player:opp_player}, inplace=True)

#Initialize new features
df_team ["team_goal_scored"] =0
df_team ["team_goal_conceded"] =0
df_team ["team_win"] =0
df_team ["team_loss"] =0
df_team["draw"] =0



for index, row in df_team.iterrows():
  
# etting Team and Opposition players
  for num in range(1, 12):

    team_player_or = "team_player_" + str(num)
    opp_player_or = "opp_player_" + str(num)
    if df_team.at[index, 'home_team_api_id'] == team_id:
      team_player_act =   df_team.at[index, team_player_or]
      opp_player_act =   df_team.at[index, opp_player_or]
    else:
      opp_player_act =   df_team.at[index, team_player_or]
      team_player_act =   df_team.at[index, opp_player_or]  

    df_team.at[index, team_player_or]  = team_player_act
    df_team.at[index, opp_player_or] = opp_player_act

  #Calculate Goals and Wins

  if df_team.at[index, 'home_team_api_id'] == team_id:
    team_goal_scored = df_team.at[index, "home_team_goal"]
    team_goal_conceded = df_team.at[index, "away_team_goal"]

  else:
    team_goal_scored = df_team.at[index, "away_team_goal"]
    team_goal_conceded = df_team.at[index, "home_team_goal"]

  df_team.at[index, 'team_goal_scored'] = team_goal_scored
  df_team.at[index, 'team_goal_conceded'] = team_goal_conceded

  goal_diff = team_goal_scored - team_goal_conceded

  if goal_diff > 0:
    df_team.at[index, "team_win"] = 1
  elif goal_diff < 0:
    df_team.at[index, "team_loss"] = 1
  else:
    df_team.at[index, "draw"] = 1


print (df_team)

#Saving outcome to a new csv file
df_team.to_csv("team_games.csv")