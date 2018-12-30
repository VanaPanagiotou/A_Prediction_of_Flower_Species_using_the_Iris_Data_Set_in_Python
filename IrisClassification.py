# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 17:55:33 2018

@author: Vana
"""

##      Importing the libraries

# numpy
import numpy as np
print('np: {}'.format(np.__version__))
# seaborn
import seaborn as sns
print('sns: {}'.format(sns.__version__))
# pandas
import pandas as pd
print('pd: {}'.format(pd.__version__))
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# matplotlib
import matplotlib.pyplot as plt
# print('plt: {}'.format(plt.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))


##      Importing the dataset

# Add meaningful column names
names = ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width', 'Species']
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
dataset = pd.read_csv(url, names=names)
dataset.head()


##      Exploratory Analysis

# Find how many instances (rows) and how many attributes (columns) the data contains

# shape
print(dataset.shape)
# (150, 5)

# more info on the data
print(dataset.info())

#<class 'pandas.core.frame.DataFrame'>
#RangeIndex: 150 entries, 0 to 149
#Data columns (total 5 columns):
#Sepal.Length    150 non-null float64
#Sepal.Width     150 non-null float64
#Petal.Length    150 non-null float64
#Petal.Width     150 non-null float64
#Species         150 non-null object
#dtypes: float64(4), object(1)
#memory usage: 5.9+ KB
#None


# Statistical Summary
# We can take a look at a summary of each attribute.
# This includes the count, mean, the min and max values and some percentiles.

print(dataset.describe())

#       Sepal.Length  Sepal.Width  Petal.Length  Petal.Width
#count    150.000000   150.000000    150.000000   150.000000
#mean       5.843333     3.054000      3.758667     1.198667
#std        0.828066     0.433594      1.764420     0.763161
#min        4.300000     2.000000      1.000000     0.100000
#25%        5.100000     2.800000      1.600000     0.300000
#50%        5.800000     3.000000      4.350000     1.300000
#75%        6.400000     3.300000      5.100000     1.800000
#max        7.900000     4.400000      6.900000     2.500000

# We can see that all of the numerical values have the same scale (centimeters) and similar ranges.


# Class Distribution: number of instances (rows) that belong to each class
print(dataset.groupby('Species').size())

#Species
#Iris-setosa        50
#Iris-versicolor    50
#Iris-virginica     50
#dtype: int64

# Class Distribution: Percentage
print(dataset.groupby('Species').size().apply(lambda x: float(x) / dataset.groupby('Species').size().sum()*100))

#Species
#Iris-setosa        33.333333
#Iris-versicolor    33.333333
#Iris-virginica     33.333333
#dtype: float64

print(pd.DataFrame(data = {'freq': dataset.groupby('Species').size(), 'percentage':dataset.groupby('Species').size().apply(lambda x: float(x) / dataset.groupby('Species').size().sum()*100)}))
#                 freq  percentage
#Species                          
#Iris-setosa        50   33.333333
#Iris-versicolor    50   33.333333
#Iris-virginica     50   33.333333

# We can see that each class has the same number of instances (50 or 33.3% of the dataset).



##      Visualizations

# Univariate Plots

# Box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False,figsize=(10,10))
plt.show()

# We can also create a histogram of each input variable to get an idea of the distribution
dataset.hist(edgecolor='black',figsize=(10,10))
plt.show()


# As we can see, two of the input variables (i.e., Sepal.Length and Sepal.Width) have a Gaussian 
# distribution. This is useful to notice as we can use algorithms that can exploit this assumption.


# Boxplot on each attribute split out by Species
dataset.boxplot(by="Species",figsize=(10,10))
plt.show()

# Setosa seems to be the most distinguishable of the three species with respect to both petal and 
# sepal attributes.


# Violinplots for each species

fig = plt.figure(figsize=(10,10))

plt.subplot(2, 2, 1)
sns.violinplot(data=dataset,x="Species", y="Petal.Length")

plt.subplot(2, 2, 2)
sns.violinplot(data=dataset,x="Species", y="Petal.Width")

plt.subplot(2, 2, 3)
sns.violinplot(data=dataset,x="Species", y="Sepal.Length")

plt.subplot(2, 2, 4)
sns.violinplot(data=dataset,x="Species", y="Sepal.Width")

plt.show()



# Multivariate Plots

# Scatter plot matrix of all pairs of attributes
from pandas.plotting import scatter_matrix
scatter_matrix(dataset,figsize=(10,10))
plt.show()

# From the figure, we notice the diagonal grouping of some pairs of attributes, which implies a high
# correlation and a predictable relationship.


# Using seaborn pairplot to see the bivariate relation between each pair of attributes

sns.pairplot(dataset, hue="Species", height=3)
plt.show()

# As we can see from the plot, the species Setosa seems to be the most distinguishable of the three 
# species with respect to both petal and sepal attributes.


##      Classification

# Create training/test sets 
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
# Create a partition (80% training, 20% testing)
validation_size = 0.20
seed = 1000
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=validation_size, random_state=seed)



# K-Means Clustering

# Since we know there are 3 classes, the optimum number of clusters is 3.
# However, we will run the elbow method, which gives the optimum number of clusters (K) for classification,
# when the number of clusters is unknown.

#Finding the optimum number of clusters for K-Means classification
from sklearn.cluster import KMeans

wcss = []
seed = 1000

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 20, random_state = seed)
    # we specify n_init as 20 to run the algorithm 20 times with 20 random starting sets of centroids 
    # and then pick the best of those 20
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)


# Plotting the 'elbow' curve
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters (K)')
plt.ylabel('WCSS') # Within Cluster Sum of Squares
plt.show()

# The x-axis is the K value and the y axis is the objective function, which is commonly defined as the 
# average distance between the datapoints and the nearest centroid.
# The optimum value for K is where the "elbow" occurs. This is when the Within Cluster Sum of Squares (WCSS)
# doesn't decrease significantly with every iteration. Adding more clusters after this point will not 
# add significant value to the classification.

# K-Means Classifier
kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 20, random_state = seed)
kmeans_Cluster = kmeans.fit_predict(X)
labels = kmeans.labels_


# Visualising the clusters

# 3D Plot
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(1, figsize=(10,10))
ax = Axes3D(fig, rect=[0, 0, 0.95, 1])
ax.scatter(X[:, 3], X[:, 0], X[:, 2],
          c=labels.astype(np.float), edgecolor="k", s=40)
ax.set_xlabel("Petal Width")
ax.set_ylabel("Sepal Length")
ax.set_zlabel("Petal Length")
plt.title("K-Means Clustering")


# 2D plots

# Create a colormap
colormap = np.array(['red', 'green', 'black'])

# Convert Species names to labels (0,1,2), 0: Iris-virginica , 1: Iris-versicolor, 2: Iris-setosa
d = {ni: indi for indi, ni in enumerate(set(dataset['Species']))}
numbers = [d[ni] for ni in dataset['Species']]

## Create numeric classes for species (0,1,2) , 0: Iris-virginica , 1: Iris-versicolor, 2: Iris-setosa
#dataset2=dataset
#dataset2.loc[dataset2['Species']=='Iris-virginica','Species']=0
#dataset2.loc[dataset2['Species']=='Iris-versicolor','Species']=1
#dataset2.loc[dataset2['Species']=='Iris-setosa','Species'] = 2

print(numbers, kmeans.labels_)

# At the predicted labels, we have to convert all the 1s to 2s and 2s to 1s for consistency
pred_labels = np.choose(kmeans.labels_, [0, 2, 1]).astype(np.int64)
print (kmeans.labels_)
print(pred_labels)

# Classification based on the Petal attributes 

plt.figure(figsize=(10,10))
# Plot the Original Classifications
plt.subplot(1, 2, 1)
plt.scatter(dataset['Petal.Length'], dataset['Petal.Width'], c=colormap[numbers], s=40)
plt.title('Real Classification')

# Plot the KMeans Classifications
plt.subplot(1, 2, 2)
plt.scatter(dataset['Petal.Length'], dataset['Petal.Width'], c=colormap[pred_labels], s=40)
plt.title('K-Means Classification')
 
 
# Classification based on the Sepal attributes 

plt.figure(figsize=(10,10))
# Plot the Original Classifications
plt.subplot(1, 2, 1)
plt.scatter(dataset['Sepal.Length'], dataset['Sepal.Width'], c=colormap[numbers], s=40)
plt.title('Real Classification')

# Plot the KMeans Classifications
plt.subplot(1, 2, 2)
plt.scatter(dataset['Sepal.Length'], dataset['Sepal.Width'], c=colormap[pred_labels], s=40)
plt.title('K-Means Classification')
 
 

# Importing metrics for evaluation
import sklearn.metrics as sm

# Accuracy
sm.accuracy_score(numbers, pred_labels)
# 0.8933333333333333

# Confusion Matrix
sm.confusion_matrix(numbers, pred_labels)
#array([[36, 14,  0],
#       [ 2, 48,  0],
#       [ 0,  0, 50]], dtype=int64)

pd.crosstab(dataset['Species'],pred_labels)
#col_0             0   1   2
#Species                    
#Iris-setosa       0   0  50
#Iris-versicolor   2  48   0
#Iris-virginica   36  14   0

# All 0 classes were correctly identified as 0
# 48 class 1 were correctly identified as 1, but 2 class 1 were missclassified as class 2
# 36 class 2 were correctly identified as 2, but 14 class 2 were missclassified as class 1


# We will now build some classification models.

# Test options 

# We will use 10-fold cross validation to estimate accuracy. 
# 10-fold cross-validation splits the dataset into 10 parts (9 parts for training and 1 part for test) 
# and repeats the train/test split 10 times.
num_folds = 10
seed = 1000
scoring = 'accuracy'


# We will evaluate 9 different models: Naive Bayes (NB), Logistic Regression (LR), 
# Linear Discriminant Analysis (LDA), Decision Tree (CART), Random Forest (RF), Gradient Boosting Method (GBM), 
# k-Nearest Neighbor (kNN), Support Vector Machine (SVM) and Neural Network (NN)

# Linear methods: LR and LDA
# Non-linear methods: NB, CART, RF, GBM, kNN, SVM and NN

models = []
# Naive Bayes
from sklearn.naive_bayes import GaussianNB
models.append(('NB', GaussianNB()))
# Logistic Regression
from sklearn.linear_model import LogisticRegression
models.append(('LR', LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=200)))
# Linear Discriminant Analysis (LDA)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
models.append(('LDA', LinearDiscriminantAnalysis()))
# Decision Tree
from sklearn.tree import DecisionTreeClassifier
models.append(('CART', DecisionTreeClassifier()))
# Random Forest
from sklearn.ensemble import RandomForestClassifier
models.append(('RF', RandomForestClassifier(n_estimators=100)))
# Gradient Boosting Method (GBM)
from sklearn.ensemble import GradientBoostingClassifier
models.append(('GBM', GradientBoostingClassifier()))
# k-Nearest Neighbour (kNN)
from sklearn.neighbors import KNeighborsClassifier
models.append(('KNN', KNeighborsClassifier()))
# Support Vector Machine (SVM)
from sklearn.svm import SVC
models.append(('SVM', SVC(gamma='scale')))
# Neural Network (NN)
from sklearn.neural_network import MLPClassifier  
models.append(('NN', MLPClassifier(hidden_layer_sizes=(10),solver='sgd',learning_rate_init=0.01,max_iter=500)))



from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# evaluate each model
results = []
names = []
for name, model in models:
	kfold = KFold(n_splits=num_folds, random_state=seed)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	model_summary = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(model_summary)

#NB: 0.958333 (0.076830)
#LR: 0.966667 (0.040825)
#LDA: 0.983333 (0.033333)
#CART: 0.950000 (0.066667)
#RF: 0.958333 (0.055902)
#GBM: 0.950000 (0.055277)
#KNN: 0.958333 (0.055902)
#SVM: 0.966667 (0.040825)
#NN: 0.983333 (0.033333)
    
# We can see that LDA and NN have the largest estimated accuracy score  
    


# We can also create a plot of the model evaluation results and compare the spread and the mean accuracy of 
# each model. There is a population of accuracy measures for each algorithm because each algorithm was evaluated
# 10 times (10-fold cross validation).

fig = plt.figure()
fig.suptitle('Models Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# Performance of models on the test set

for name, model in models:
    md = model
    md.fit(X_train, Y_train)
    predictions = md.predict(X_test)
    print('The accuracy of the' , name, 'classifier on test data is {:.4f}'.format(sm.accuracy_score(Y_test, predictions)))
    print('\n')
    print('The confusion matrix of the' , name, 'classifier using test data is:\n')
    print(sm.confusion_matrix(Y_test, predictions))
    print('\n')
    print('The classification report of the' , name, 'classifier using test data is:\n')
    print(sm.classification_report(Y_test, predictions))
    print('\n')


# Accuracy
#NB : 0.9333
#LR : 0.9333
#LDA : 0.9667
#CART : 0.9000
#RF : 0.9333
#GBM : 0.9333
#KNN : 0.9667
#SVM : 0.9667
#NN : 0.9667

# We can see that the LDA and NN algorithms have the largest accuracy on the test set as well, along with
# kNN and SVM algorithms.
