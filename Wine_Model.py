import pandas as pd
import numpy as np

WineDataset = pd.read_csv("../WineDataset.csv", encoding = "ISO-8859-1")

#The goal of TextBlob is to gather the Polarity and Subjectivity of each description.
#Polarity has a range of [-1,1] and measures how positive or negative the description isself.
#Subjectivity has a range of [0,1] and measures how objective (0) or subjective (1) each description is.
#https://textblob.readthedocs.io/en/dev/quickstart.html#create-a-textblob
from textblob import TextBlob
from textblob.tokenizers import WordTokenizer
tokenizer = WordTokenizer()
WineDataset['Polarity'] = WineDataset["description"].apply(lambda text: TextBlob(text, tokenizer=tokenizer).polarity)
WineDataset['Subjectivity'] = WineDataset["description"].apply(lambda text: TextBlob(text, tokenizer=tokenizer).subjectivity)

#This model will contain each wines Polarity, Subjectivity, Price, Country of Origin, and the target will be
#the amount of points it recieved.
WineDataset_ = WineDataset.loc[:,["Polarity", "Subjectivity", "price", "points", "country"]]
columns = ["country"]
WineDataset_ = pd.get_dummies(WineDataset_, columns=columns)
WineDataset = WineDataset_.dropna()

X = WineDataset.drop(columns="points", axis = 1)
Y = WineDataset["points"]

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X,Y, test_size=0.30, random_state = 123)

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

#http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
Tree = DecisionTreeRegressor(max_features= 'auto')
max_depth = [10,11,12,13,14,15]
min_samples_leaf = [1,5]
param_grid = dict(max_depth = max_depth, min_samples_leaf = min_samples_leaf)

GridSearch = GridSearchCV(estimator=Tree, param_grid=param_grid, cv=5, n_jobs=2, scoring='neg_mean_squared_error')
GridResults = GridSearch.fit(xtrain,ytrain)
print("Best: {} using: {}".format(GridResults.best_score_, GridResults.best_params_))

#http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
Forest = RandomForestRegressor(max_depth=10, min_samples_leaf=5, n_estimators=1000)
Forest.fit(xtrain,ytrain)
Predictions = Forest.predict(xtest)

from sklearn.metrics import mean_squared_error, r2_score
print('MSE: {} and R2: {}'.format(mean_squared_error(y_true=ytest, y_pred=Predictions), r2_score(ytest, Predictions)))
