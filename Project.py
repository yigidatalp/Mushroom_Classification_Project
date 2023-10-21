# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 12:05:13 2023

@author: Yigitalp
"""
# import data wrangling libraries
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from xgboost import plot_tree
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# import data visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

# import encoding, splitting, and feature selection libraries

# import classifer libraries

# import metrics libraries

# import the dataset
df = pd.read_csv('mushrooms.csv')

# check the dataset: Since veil-type has 1 unique value, it can be dropped
df.info()
df.head()
df.nunique()
df = df.drop('veil-type', axis=1)

# Encode target
class_value_counts = df['class'].value_counts()
df['class'] = df['class'].replace(
    [class_value_counts.idxmin(), class_value_counts.idxmax()], [0, 1])

# Grouper function for plotting and encoding


def grouper(feature):
    group_feature_class = df.groupby([feature, 'class'])['class'].count()
    group_feature_class = group_feature_class.rename('count')
    group_feature_class = group_feature_class.reset_index()
    hue_order = group_feature_class.sort_values(['class', 'count'], ascending=[
                                                True, False])[feature].unique()
    return group_feature_class, hue_order

# Plotter function for EDA with bivariate analysis


def plotter(feature):
    group_feature_class, hue_order = grouper(feature)
    fig, ax = plt.subplots()
    sns.barplot(data=group_feature_class, x='class', y='count',
                hue=feature, hue_order=hue_order, ax=ax)
    plt.ylabel('count')
    for label in ax.containers:
        ax.bar_label(label,)

# Encoder function with MinMax Scaler


def target_encoder(feature):
    group_feature_class, _ = grouper(feature)
    group_feature_class['score'] = group_feature_class['class'] * \
        group_feature_class['count']
    group_feature_class = group_feature_class.groupby(feature).agg(
        sum_of_counts=('count', np.sum),
        sum_of_scores=('score', np.sum)
    )
    group_feature_class = group_feature_class.reset_index()
    group_feature_class['score'] = group_feature_class['sum_of_scores'] / \
        group_feature_class['sum_of_counts']
    code_list = [code for code in range(len(group_feature_class))]
    sorted_group = group_feature_class.sort_values(by='score')
    df[feature] = df[feature].replace(list(sorted_group[feature]), code_list)
    df[feature] = (df[feature] - df[feature].min()) / \
        (df[feature].max() - df[feature].min())


# Encode all features
for col in df.columns:
    if col != 'class':
        plotter(col)
        target_encoder(col)

# Optional
"""
# Feature selection: Correlation
corr = abs(df.corr())
corr = corr['class']
corr = corr.drop(index='class')
corr.name = 'Corr_scores'
"""

# Create X and y
X = df.drop('class', axis=1)
y = df['class']

# Optional
"""
# Feature selection: MI score
MI_score = pd.Series(mutual_info_classif(X, y), name='MI_scores')
MI_score.index = X.columns

# Feature selection per weighted score
feature_scores = pd.concat([corr, MI_score], axis=1, names=[
                           'Corr_scores', 'MI_scores'])
for col in feature_scores:
    feature_scores[col] = (feature_scores[col]-feature_scores[col].min()) / \
        (feature_scores[col].max()-feature_scores[col].min())
alpha = 0.5
feature_scores['Weighted_scores'] = alpha * \
    feature_scores['Corr_scores']+(1-alpha)*(feature_scores['MI_scores'])
selected_features = [
    col for col in feature_scores.index if feature_scores.loc[col, 'Weighted_scores'] >= alpha]
X = X[selected_features]
"""

# Split dataset into train and test sets per 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Create classifier models
logistic_classifer = LogisticRegression()
ridge_classifier = RidgeClassifier()
decision_tree_classifier = DecisionTreeClassifier()
naive_bayes_classifier = GaussianNB()
neural_network_classifier = MLPClassifier()
random_forest_classifier = RandomForestClassifier()
xgboost_classifier = XGBClassifier()

# Train classifier models
logistic_classifer.fit(X_train, y_train)
ridge_classifier.fit(X_train, y_train)
decision_tree_classifier.fit(X_train, y_train)
naive_bayes_classifier.fit(X_train, y_train)
neural_network_classifier.fit(X_train, y_train)
random_forest_classifier.fit(X_train, y_train)
xgboost_classifier.fit(X_train, y_train)

# Make predictions
logistic_predictions = logistic_classifer.predict(X_test)
ridge_predictions = ridge_classifier.predict(X_test)
decision_tree_predictions = decision_tree_classifier.predict(X_test)
naive_bayes_predictions = naive_bayes_classifier.predict(X_test)
neural_network_predictions = neural_network_classifier.predict(X_test)
random_forest_predictions = random_forest_classifier.predict(X_test)
xgboost_predictions = xgboost_classifier.predict(X_test)

# Evaluate performance metrics
logistic_report = classification_report(y_test, logistic_predictions)
ridge_report = classification_report(y_test, ridge_predictions)
decision_tree_report = classification_report(y_test, decision_tree_predictions)
naive_bayes_report = classification_report(y_test, naive_bayes_predictions)
neural_network_report = classification_report(
    y_test, neural_network_predictions)
random_forest_report = classification_report(y_test, random_forest_predictions)
xgboost_report = classification_report(y_test, xgboost_predictions)

# Print metrics report: Other than Ridge and Naive Bayes Classifiers, 100% accuracy score is achieved
print('Logistic Regression')
print(logistic_report)
print(confusion_matrix(y_test, logistic_predictions))
print()
print('Ridge Classifier')
print(ridge_report)
print(confusion_matrix(y_test, ridge_predictions))
print()
print('Decision Tree')
print(decision_tree_report)
print(confusion_matrix(y_test, decision_tree_predictions))
print()
print('Naive Bayes')
print(naive_bayes_report)
print(confusion_matrix(y_test, naive_bayes_predictions))
print()
print('Neural Network')
print(neural_network_report)
print(confusion_matrix(y_test, neural_network_predictions))
print()
print('Random Forest')
print(random_forest_report)
print(confusion_matrix(y_test, random_forest_predictions))
print()
print('XGBoost Classifier')
print(xgboost_report)
print(confusion_matrix(y_test, xgboost_predictions))
print()
fig, ax = plt.subplots(figsize=(15, 10))
plot_tree(xgboost_classifier, ax=ax)
plt.savefig('XGBoost Decision Tree.pdf')
