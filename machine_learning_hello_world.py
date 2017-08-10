'''
Machine learning attempts to build a data model based on features of the data, for example did the passenger
have a first, second or third class ticket, was the passenger male or female and so on. The model can then
be used to make predictions or classify a data point. In this case I'm using the Titanic passenger list, the list is divided
into two parts - part one is the complete data including a binary field which indicates if the passenger survived or not. The
second part has this field removed, each machine learning algorithm attempts to predict if the passengers in this list live or die then 
compares with the actual result for the passenger and outputs a success rate - using the score method, see for example:
http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

The 'art' in supervised machine learning is adjusting the
parameters and data features used to get the best success without over fitting. It is more complicated than it seems because not all data
features are independent for eample age and passenger class (pclass) are related - first class passengers tend to older than 2nd class who are older than 3rd class
The accompanying repo https://github.com/ponll/titanic_data_analysis includes a .txt file with some data analysis on passenger survivability


this problem is a classification problem as every passenger fell into one of two categories - they survived or they died
so the 3 machine learning techniques used below are classification techniques, they are also supervised
learning techniques which means we need to break the original data set into a training dataset and a test
dataset. The difference between classification and regression:

Regression: the output variable takes continuous values.

Classification: the output variable takes class labels.


The original data file is available from various sources, for example: http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic3.xls
After downloading the file the only change I made was I converted it to a csv file.

The script was written and tested using idle in python 2 but it should be compatible with both python 2 and 3
'''

#import the necessary libraries
import pandas as pd
from sklearn import tree, preprocessing
import sklearn.ensemble as ske
from sklearn.model_selection import train_test_split

#read the data into a pandas dataframe - this makes the preprocessing of the data easier
df = pd.read_csv('titanic_data.csv')
#the data needs to be prepared for ML - drop fields which have lots of missing data, then drop rows with missing data
df = df.drop(['body','cabin','boat','home.dest','name','ticket'],axis=1)
df = df.dropna()

#machine learning needs numerical values not strings
le = preprocessing.LabelEncoder()
df.sex = le.fit_transform(df.sex)
df.embarked = le.fit_transform(df.embarked)
'''
a row from the original data looked like:
pclass	survived	name	                sex	age	sibsp	parch	ticket	fare	        cabin	embarked	boat	body	home.dest
1	1	Allen, Miss. Elisabeth Walton	female	29	0	0	24160	211.3375	B5	S	          2		St Louis, MO
a typical row now looks like:
pclass  survived  sex      age      sibsp  parch      fare      embarked
1         1         0   29.0000      0      0       211.3375         2
'''
#create two new numpy arrays, X has the survived column values removed and y is only the survived column values
X = df.drop(['survived'], axis=1).values
y = df['survived'].values

#we are using supervised learning so we need training and test data, the test_size parameter determines the relative sizes of the training and test data sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=0)

#use three different approaches and print out the success rate (1 = 100%), you can vary the parameters below and the test_size parameter above to try to
#improve success rate
clf_dt = tree.DecisionTreeClassifier(max_depth=10)
clf_dt = clf_dt.fit(X_train, y_train)
print(clf_dt.score(X_test,y_test))
      
clf_rf = ske.RandomForestClassifier(n_estimators=50)
clf_rf = clf_rf.fit(X_train, y_train)
print(clf_rf.score(X_test,y_test))

clf_gb = ske.GradientBoostingClassifier(n_estimators=50)
clf_gb = clf_gb.fit(X_train, y_train)
print(clf_gb.score(X_test,y_test))

'''
I found the gradient boosting technique gave the best results (about 82% accuracy without any fine tuning) and the decision
tree gave the worst results

You'll find plenty of examples online explaining what random forest, decision trees and gradient boosting are.
'''



