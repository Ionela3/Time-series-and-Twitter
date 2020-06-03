import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from scipy import stats
from scipy.stats import skew, mode
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
import statsmodels
import xgboost as xgb
import itertools
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import statsmodels.formula.api as smf
from sklearn import metrics
##### Import train and test datasets #####
dataset = pd.read_excel('set_date_final_1.xls')
X = dataset.iloc[:, 2:3].values
y = dataset.iloc[:, 4].values




# # Train test split

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=3)

print(f'The train dataset has {X_train.shape[0]} observations')
print(f'The train dataset has {X_test.shape[0]} observations')


##### Modeling #####
# Create a cross validation function
def get_best_model(estimator, params_grid = {}):
    model = GridSearchCV(estimator = estimator, param_grid = params_grid, cv = 3, scoring = "accuracy", n_jobs = -1)
    model.fit(X_train, Y_train)
    print('\n--- Best Parameters -----------------------------')
    print(model.best_params_)
    print('\n--- Best Model -----------------------------')
    best_model = model.best_estimator_
    print(best_model)
    return best_model


# The confusion matrix plotting function is from the sklearn documentation below:
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(conf_matrix, classes, title, normalize = False, cmap = plt.cm.Purples):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(conf_matrix, interpolation = 'nearest', cmap =  cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)

    if normalize:
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis = 1)[:, np.newaxis]

    thresh = conf_matrix.max() / 2.
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(j, i, conf_matrix[i, j],
                 horizontalalignment="center",
                 color="black" if conf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

class_names = ['Success', 'Failure']


# Create a model fitting function
def model_fit(model, feature_imp, cv = 5):
    # model fit
    clf = model.fit(X_train, Y_train)

    # model prediction
    y_pred = clf.predict(X_test)

    # model report
    conf_matrix = confusion_matrix(Y_test, y_pred)
    plot_confusion_matrix(conf_matrix, classes = class_names, title = 'Confusion matrix')


    print('\n--- Train Set -----------------------------')
    print('Accuracy: %.5f +/- %.4f' % (np.mean(cross_val_score(clf, X_train, Y_train, cv = cv)), np.std(cross_val_score(clf, X_train, Y_train, cv = cv))))
    print('AUC: %.5f +/- %.4f' % (np.mean(cross_val_score(clf, X_train, Y_train, cv = cv, scoring = 'roc_auc')), np.std(cross_val_score(clf, X_train, Y_train, cv = cv, scoring = 'roc_auc'))))
    print('\n--- Validation Set -----------------------------')
    print('Accuracy: %.5f +/- %.4f' % (np.mean(cross_val_score(clf, X_test, Y_test, cv = cv)), np.std(cross_val_score(clf, X_test, Y_test, cv = cv))))
    print('AUC: %.5f +/- %.4f' % (np.mean(cross_val_score(clf, X_test, Y_test, cv = cv, scoring = 'roc_auc')), np.std(cross_val_score(clf, X_test, Y_test, cv = cv, scoring = 'roc_auc'))))
    print('-----------------------------------------------')



# ## k-Nearest Neighbors (KNN) ##
# # An accuracy of 0.76 is not very impressive. I will just take this as the model benchmark.
# knn = KNeighborsClassifier()
# parameters = {'n_neighbors': [5, 6, 7],
#               'p': [1, 2],
#               'weights': ['uniform', 'distance']}
# clf_knn = get_best_model(knn, parameters)
# model_fit(model = clf_knn, feature_imp = False)
# # y_pred = clf_knn.predict(X_test)
# # plt.show()
#
# ## Naive Bayes Classifier ##
# # As expected, Naive Bayes classifier doesn't perform well here.
# # There are multiple reasons. Some of the numeric features are not normally distributed, which is a strong assemption hold by Naive Bayes.
# # Also, features are definitely not independent.
# clf_nb = GaussianNB()
# model_fit(model = clf_nb, feature_imp = False)
# y_pred = clf_nb.predict(X_test)
# plt.show()

# ## Logistic Regression ##
# # We're making progress here. Logistic regression performs better than KNN.
# lg = LogisticRegression(random_state = 3, max_iter = 2400)
# parameters = {'C': [0.8, 0.9, 1]}
# clf_lg = get_best_model(lg, parameters)
# model_fit(model = clf_lg, feature_imp = True)
# y_pred = clf_lg.predict(X_test)
# plt.show()
# print(np.sqrt(metrics.mean_squared_error(Y_test,y_pred)))
#
#
# # Random Forest ##
#
# rf = RandomForestClassifier(random_state = 3)
# parameters = {'n_estimators': [100],
#               'max_depth': [10],
#               'min_samples_split': [3]}
# clf_rf= get_best_model(rf, parameters)
# model_fit(model = clf_rf, feature_imp = True)
#
# # # # model prediction
# # y_pred = clf_rf.predict(X_test)
# # plt.show()
#
# ## Support Vector Machines ##
# # try a SVM RBF model
# svc = svm.SVC(kernel = 'rbf', probability = True, random_state = 3)
# parameters = {'gamma': [0.0001, 0.001, 0.005, 0.1, 1, 3, 5],
#               'C': [0.5, 1, 5, 0.1, 100, 1000]}
# clf_svc = get_best_model(svc, parameters)
# model_fit(model = clf_svc, feature_imp = False)
# # y_pred = clf_svc.predict(X_test)
# # plt.show()
# #
# # print(clf_svc.n_support_)
#
# ## XGBoost ##
#
# xgb = xgb.XGBClassifier()
# parameters = {'n_estimators': [900, 1000, 1100],
#               'learning_rate': [0.01],
#               'max_depth': [8],
#               'min_child_weight': [1],
#               'subsample': [0.8],
#               'colsample_bytree': [0.3, 0.4, 0.5]}
# clf_xgb = get_best_model(xgb, parameters)
# model_fit(model = clf_xgb, feature_imp = True)
# # y_pred = clf_xgb.predict(X_test)
# # plt.show()
#
# ## Adaptative Boosting (adaboost) ##
# ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200)
# clf_ada = get_best_model(ada)
# model_fit(model = clf_ada, feature_imp = True)
# ## model prediction
# # y_pred = clf_ada.predict(X_test)
# # plt.show()



# ##### Model Evaluation #####
# # Compare model performance
# classifiers = [clf_knn, clf_nb, clf_lg, clf_rf, clf_svc, clf_xgb,clf_ada]
# index = ['K-Nearest Neighbors', 'Naive Bayes', 'Logistic Regression', 'Random Forest', 'Support Vector Machines', 'XGBoost', 'AdaBoost']
# scores = []
# for clf in classifiers:
#     score = np.mean(cross_val_score(clf, X_test, Y_test, cv = 5, scoring = 'accuracy'))
#     scores = np.append(scores, score)
# models = pd.Series(scores, index = index)
# models.sort_values(ascending = True, inplace = True)
#
# print(models)