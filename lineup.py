import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

class DynamicLabelEncoder:
  
  def __init__(self):
    self.fresh_id = 0
    self.mapping = {} 
    
  def convert_item (self, arg_item):
    
    if arg_item in self.mapping:
      return self.mapping[arg_item]
    else:
      used_id =  self.fresh_id
      self.fresh_id += 1

      self.mapping[arg_item] = used_id
      return used_id
    
  def convert_list(self, arg_list):
    
    return [self.convert_item(item) for item in arg_list] 
  
  def inverse_convert(self, item):
    pass
    
        
enc = DynamicLabelEncoder()

dataset = pd.read_csv('team_games.csv')
  #print(dataset.loc[0, 'team_player_1'])
  
match_to_players = [] # [match_id, list of short player ids]
match_to_team_win = [] # [match_idx, 1]

goal_difference = [] # [goal_difference, 1]

for match_id, row in dataset.iterrows():
  
  match_short_ids = []
  for col_name_idx in range(1, 11+1):
    header = "team_player_{}".format(col_name_idx)
    value = row[header]
    if math.isnan(value):
      value = -1
    long_player_id = int(value)
    
    short_player_id = enc.convert_item(long_player_id)
    match_short_ids.append(short_player_id)
  
  #goal_difference.append(dataset.at[(match_id, "team_goal_scored")] - dataset.at[(match_id, "team_goal_conceded")])
  
  match_to_players.append(match_short_ids)
  match_to_team_win.append([row['team_win']])
    

from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
many_hot_vectors = mlb.fit_transform(match_to_players)

#assert that each many_hot vectors sums to 11
np.testing.assert_allclose(np.sum(many_hot_vectors, axis=1), np.ones((many_hot_vectors.shape[0]))*11)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(match_to_players, match_to_team_win, test_size = 0.25, random_state = 50)

print("==TRYING Decision Trees==")

from sklearn.tree import DecisionTreeClassifier
#clf = DecisionTreeClassifier()
clf = DecisionTreeClassifier(random_state=0, min_samples_leaf=10)

clf.fit(X_train, y_train)

test_predictions = clf.predict(X_test)
train_predictions = clf.predict(X_train)


from sklearn import metrics

print("Train accuracy: ")
print(metrics.accuracy_score(y_train, train_predictions))

print("Test accuracy:")
print(metrics.accuracy_score(y_test, test_predictions))

#test_probs = clf.predict_proba(X_test)

from sklearn import tree

with open('barca_player_decision_tree.dot', 'w') as f:
  tree.export_graphviz(clf, out_file=f, filled=True,
                       feature_names=["player_{}".format(i) for i in range(1,len(match_to_players[0])+1)])
  
#view in http://webgraphviz.com/


print("==TRYING RANDOM FORREST==")
from sklearn.ensemble import RandomForestClassifier
#rfclf = RandomForestClassifier()
rfclf = RandomForestClassifier(n_estimators=20, min_samples_leaf=4, criterion="entropy", random_state=3)
rfclf.fit(X_train, y_train)
test_predictions = rfclf.predict(X_test)
train_predictions = rfclf.predict(X_train)
print("Train accuracy: ")
print(metrics.accuracy_score(y_train, train_predictions))

print("Test accuracy:")
print(metrics.accuracy_score(y_test, test_predictions))

print("==TRYING SVM==")
from sklearn.svm import SVC
#svc = SVC()
svc = SVC(C=1,kernel='rbf', degree=3, random_state=3, gamma='auto')
svc.fit(X_train, y_train)
test_predictions = svc.predict(X_test)
train_predictions = svc.predict(X_train)
print("Train accuracy: ")
print(metrics.accuracy_score(y_train, train_predictions))

print("Test accuracy:")
print(metrics.accuracy_score(y_test, test_predictions))

 

print("==TRYING DNN==")
from sklearn.neural_network import MLPClassifier
mlpclf = MLPClassifier(hidden_layer_sizes=(5, 3,3,3,3), batch_size=32, 
                       learning_rate='adaptive', learning_rate_init=1e-1,
                       early_stopping=True, random_state=3)
mlpclf.fit(X_train, y_train)
test_predictions = mlpclf.predict(X_test)
train_predictions = mlpclf.predict(X_train)
print("Train accuracy: ")
print(metrics.accuracy_score(y_train, train_predictions))

print("Test accuracy:")
print(metrics.accuracy_score(y_test, test_predictions))

import matplotlib.pyplot as plt

print("==TRAIN PROB HISTOGRAM==")
for clfname, classifier in zip(["decision tree", "random forrest", "dnn"], [clf,rfclf,mlpclf]):
  p = np.array(classifier.predict_proba(X_train))[:,1]
  weights = np.ones_like(p)/float(len(p))
  plt.hist(p, bins=20, range=(0.0,1.0), weights=weights, alpha=0.5, label=clfname)
  plt.legend(loc='upper left')
  
plt.savefig('Train Prob Histogram')
plt.figure()
print("==TEST PROB HISTOGRAM==")
for clfname, classifier in zip(["decision tree", "random forrest", "dnn"], [clf,rfclf,mlpclf]):
  p = np.array(classifier.predict_proba(X_test))[:,1]
  weights = np.ones_like(p)/float(len(p))
  plt.hist(p, bins=20, range=(0.0,1.0), weights=weights, alpha=0.5, label=clfname)
  plt.legend(loc='upper left')
plt.savefig('Test Prob Histogram')


print("Agreement between random forrest and DNN:")
print(np.sum(mlpclf.predict(X_test)==rfclf.predict(X_test)) / float(len(X_test)))

