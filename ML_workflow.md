Model: XGB

1. Collect Data
	Features, Labels
2. Prep
	1. Imputation - .fillna(mean, 0.0)
	1.5. Identify the outliers, look at them personally
	2. One-Hot Encoding -> Textual features
	3. Drop useless columns
	3.5. Normalization of large range features
	4. Apply unsupervised learning techniques to generate pseudo-labels / new features (second pass)
	4. Labels
3. Split data
	1. Data --> Features and Labels
	2. Features and Labels --> Training and testing features and labels
4. Train the model --> XGB / SVM / LogiReg / Random Forests / Neural Networks
	4.1. Ensemble Learning - Bagging, Boosting, Voting
5. Optimize the model using Hyperparameter tuning --> GridSearchCV
6. You make inferences

7. Explain the feature relations --> SHAP

cupy, sklearn