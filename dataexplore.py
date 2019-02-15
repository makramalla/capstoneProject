import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from pprint import pprint


dataset = pd.read_csv('team_games.csv')


########################Predicting Outcome based on Goals and Possesssion #############################

##Pssession##
import xml.etree.ElementTree as ET

possession_xmls = dataset.iloc[:, 85]
possessions = []

for _xml in possession_xmls:
  root = ET.fromstring(_xml)
  if len(root) == 0:
    poss_value = -1
  else:
    poss_value = float(root[-1][0].text)/100.0
    
  possessions.append(poss_value)

np_poss = np.array(possessions)
npmask = np_poss == -1
poss_avg = np.average(np_poss, weights = np.where(npmask, np.zeros_like(np_poss), np.ones_like(np_poss) ) )

np.place(np_poss , npmask, poss_avg)

np_poss = np.expand_dims(np_poss , axis=1)


##Goal Scored##


all_goals_scored =  dataset.loc[: ,['team_goal_scored']].values

print(np_poss.shape)
print(all_goals_scored.shape)


##Combine Features##

all_features = np.concatenate( (all_goals_scored , np_poss) , axis= 1)
print(all_features.shape)
#print(all_features)



X_tree = all_features
y_tree = dataset.loc[:, 'team_win'].values

from sklearn import tree

# Splitting the dataset into the Training set and Test set
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
X_tree_train, X_tree_test, y_tree_train, y_tree_test = train_test_split(X_tree, y_tree, test_size = 0.25, random_state = 54)

clf = DecisionTreeClassifier(criterion = 'entropy', min_samples_leaf=3)

clf.fit(X_tree_train, y_tree_train )


# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_tree_train, y_tree_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min(), stop = X_set[:, 0].max() + 0.3, step = 0.01),
                     np.arange(start = X_set[:, 1].min(), stop = X_set[:, 1].max() + 0.3, step = 0.01))
plt.contourf(X1, X2, clf.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Decision Tree Classification (Training set)')
plt.xlabel('Goals')
plt.ylabel('Possession')
plt.legend()
plt.savefig('Goals and possession - Training Set.png')

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_tree_test, y_tree_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min(), stop = X_set[:, 0].max() + 0.3, step = 0.01),
                     np.arange(start = X_set[:, 1].min(), stop = X_set[:, 1].max() + 0.3, step = 0.01))
plt.contourf(X1, X2, clf.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Decision Tree Classification (Test set)')
plt.xlabel('Goals')
plt.ylabel('Possession')
plt.legend()
plt.savefig('Goals and possession - Testing Set.png')




######################## Predicting Outcome based on Goals and Stage #############################

##Goal Scored##

goals_scored =  dataset.loc[: ,['team_goal_scored']].values

##Stage##

stage =  dataset.loc[: ,['stage']].values

print(stage.shape)
print(goals_scored.shape)


##Combine Features##

all_features_2 = np.concatenate( (all_goals_scored , stage) , axis= 1)
print(all_features_2.shape)
#print(all_features)


X_tree_2 = all_features_2
y_tree_2 = dataset.loc[:, 'team_win'].values

X_tree_train, X_tree_test, y_tree_train, y_tree_test = train_test_split(X_tree_2, y_tree_2, test_size = 0.25, random_state = 25)

clf_2 = DecisionTreeClassifier(criterion = 'entropy', min_samples_leaf=7)

clf_2.fit(X_tree_train, y_tree_train )

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_tree_train, y_tree_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min(), stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min(), stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, clf_2.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Decision Tree Classification (Training set)')
plt.xlabel('Goals')
plt.ylabel('stage')
plt.legend()
plt.savefig('Goals and stage - Training Set.png')

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_tree_test, y_tree_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min(), stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min(), stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, clf_2.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Decision Tree Classification (Test set)')
plt.xlabel('Goals')
plt.ylabel('Stage')
plt.legend()
plt.savefig('Goals and stage - Testing Set.png')



######################## Predicting Outcome based on Goals scored and Goals Conceded (Just for fun) #############################


##Goal Scored##


all_goals_scored =  dataset.loc[: ,['team_goal_scored']].values

print(all_goals_scored.shape)


##Goal Conceded##


all_goals_conceded =  dataset.loc[: ,['team_goal_conceded']].values

print(all_goals_conceded.shape)

##Combine Features##

all_features_3 = np.concatenate( (all_goals_scored , all_goals_conceded) , axis= 1)
print(all_features.shape)
#print(all_features)



X_tree_3 = all_features_3
y_tree_3 = dataset.loc[:, 'team_win'].values

X_tree_train, X_tree_test, y_tree_train, y_tree_test = train_test_split(X_tree_3, y_tree_3, test_size = 0.25, random_state = 25)

clf_3 = DecisionTreeClassifier(criterion = 'entropy') #, min_samples_leaf=7)

clf_3.fit(X_tree_train, y_tree_train )

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_tree_train, y_tree_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min(), stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min(), stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, clf_3.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Decision Tree Classification (Training set)')
plt.xlabel('Goals Scored')
plt.ylabel('Goals Conceded')
plt.legend()
plt.savefig('Goalsscored and conceded - Training Set.png')

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_tree_test, y_tree_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min(), stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min(), stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, clf_3.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Decision Tree Classification (Training set)')
plt.xlabel('Goals Scored')
plt.ylabel('Goals Conceded')
plt.legend()
plt.savefig('Goalsscored and conceded - Testing Set.png')