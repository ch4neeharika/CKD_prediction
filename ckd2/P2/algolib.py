import sys
import random
import csv
try:
    # Python 3
    from itertools import zip_longest
except ImportError:
    # Python 2
    from itertools import izip_longest as zip_longest

import pickle as pk
import struct
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from featureselector import FeatureSelector
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import RFE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.metrics import confusion_matrix

#########################################
# Project   : ML Algorithms toolbox     #
# Created   : 12/23/17 13:58:06         #
# Author    : Omodaka9375                #
# Licence   : MIT                       #
#########################################

algo_list = ['dtc', 'linsvc', 'svc', 'mlp', 'knn', 'gaus', 'lda', 'logreg']

export_path = './models/'

# Data processing tools

def testAlgo(path='', samples=None, algo='', export=False, log=False, standard_scaler=False, minmax_scaler=False):

    if path == '' or algo == '':
        print('You need to specify all required parameters!')
        sys.exit(0)

    print('Training started . . .')
    X, y = encodelabels(path, row_count=samples)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=111, shuffle=True)

    if standard_scaler and not minmax_scaler:
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
    elif minmax_scaler and not standard_scaler:
            mm = MinMaxScaler()
            X_train = mm.fit_transform(X_train)
            X_test = mm.transform(X_test)
    elif standard_scaler and minmax_scaler:
            print('You can only use one scaler at a time- minmax or standard!')
            sys.exit(0)

    if algo == 'dtc':
        model = DecisionTreeClassifier()
    if algo == 'linsvc':
        model = svm.LinearSVC()
    if algo == 'svc':
        model = svm.SVC()
    if algo == 'mlp':
        model = MLPClassifier()
    if algo == 'knn':
        model = KNeighborsClassifier()
    if algo == 'gaus':
        model = GaussianNB()
    if algo == 'lda':
        model = LinearDiscriminantAnalysis()
    if algo == 'logreg':
        model = LogisticRegression()

    model.fit(X_train, y_train)

    trainY = np.array(y_train)
    prediction_train = model.predict(X_train)

    testY = np.array(y_test)
    prediction_test = model.predict(X_test)

    training_accuracy = accuracy_score(trainY, prediction_train)
    test_accuracy = accuracy_score(testY, prediction_test)

    if export:
        export_trained_model(model, algo + '_classifier')
    return training_accuracy, test_accuracy


def testHypersOnAlgo(path='', samples=None, algo=[], hparameters={}, standard_scaler=False, minmax_scaler=False, folds=5, save_best=False, search='random'):

    if path == '' or len(algo) != len(hparameters) or len(hparameters) < 1:
        print('You need to specify all required parameters!')
        sys.exit(0)

    print('\n######### Hyper ' + search + ' training started #########\n')

    X, y = encodelabels(path, row_count=samples)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=111, shuffle=True)

    if standard_scaler and not minmax_scaler:
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
    elif minmax_scaler and not standard_scaler:
            mm = MinMaxScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
    elif standard_scaler and minmax_scaler:
            print('You can only use one scaler at a time- minmax or standard!')
            sys.exit(0)

    best_score = {}

    if 'dtc' in hparameters:
                tree = DecisionTreeClassifier()

                if search == 'random':
                    model = RandomizedSearchCV(tree, hparameters['dtc'], cv=folds)
                elif search == 'grid':
                    model = GridSearchCV(tree, hparameters['dtc'], cv=folds)

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                par = "Tuned DTC model parameters: {}".format(model.best_params_)
                best_score.update({par: model.best_score_})
                cm = confusion_matrix(y_test, y_pred)
                print(par)
                print('Best score: ' + str(model.best_score_*100) + ' %')
                print('DTC confusion matrix:')
                print(cm)
                if save_best:
                    export_trained_model(model, 'best_dtc')

    if 'linsvc' in hparameters:
                tree = svm.LinearSVC()
                if search == 'random':
                    model = RandomizedSearchCV(tree, hparameters['linsvc'], cv=folds)
                elif search == 'grid':
                    model = GridSearchCV(tree, hparameters['linsvc'], cv=folds)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                par = "Tuned LinearSVC model parameters: {}".format(model.best_params_)
                best_score.update({par: model.best_score_})
                cm = confusion_matrix(y_test, y_pred)
                print(par)
                print('Best score: ' + str(model.best_score_*100) + ' %')
                print('LinearSVC confusion matrix:')
                print(cm)
                if save_best:
                    export_trained_model(model, 'best_linsvc')

    if 'svc' in hparameters:
                tree = svm.SVC()
                if search == 'random':
                    model = RandomizedSearchCV(tree, hparameters['svc'], cv=folds)
                elif search == 'grid':
                    model = GridSearchCV(tree, hparameters['svc'], cv=folds)
                model.fit(X_train,y_train)
                y_pred = model.predict(X_test)
                par = "Tuned SVC model parameters: {}".format(model.best_params_)
                best_score.update({par: model.best_score_})
                cm = confusion_matrix(y_test, y_pred)
                print(par)
                print('Best score: ' + str(model.best_score_*100) + ' %')
                print('SVC confusion matrix:')
                print(cm)
                if save_best:
                    export_trained_model(model, 'best_svc')

    if 'mlp' in hparameters:
                tree =  MLPClassifier()
                if search == 'random':
                    model = RandomizedSearchCV(tree, hparameters['mlp'], cv=folds)
                elif search == 'grid':
                    model = GridSearchCV(tree, hparameters['mlp'], cv=folds)
                model.fit(X_train,y_train)
                y_pred = model.predict(X_test)
                par = "Tuned MLP model parameters: {}".format(model.best_params_)
                best_score.update({par: model.best_score_})
                cm = confusion_matrix(y_test, y_pred)
                print(par)
                print('Best score: ' + str(model.best_score_*100) + ' %')
                print('MLP confusion matrix:')
                print(cm)
                if save_best:
                    export_trained_model(model, 'best_mlp')

    if 'knn' in hparameters:
                tree = KNeighborsClassifier()
                if search == 'random':
                    model = RandomizedSearchCV(tree, hparameters['knn'], cv=folds)
                elif search == 'grid':
                    model = GridSearchCV(tree, hparameters['knn'], cv=folds)
                model.fit(X_train,y_train)
                y_pred = model.predict(X_test)
                par = "Tuned KNN model parameters: {}".format(model.best_params_)
                best_score.update({par: model.best_score_})
                cm = confusion_matrix(y_test, y_pred)
                print(par)
                print('Best score: ' + str(model.best_score_*100) + ' %')
                print('KNN confusion matrix:')
                print(cm)
                if save_best:
                    export_trained_model(model, 'knn_dtc')

    if 'gaus' in hparameters:
                tree = GaussianNB()
                if search == 'random':
                    model = RandomizedSearchCV(tree, hparameters['gaus'], cv=folds)
                elif search == 'grid':
                    model = GridSearchCV(tree, hparameters['gaus'], cv=folds)
                model.fit(X_train,y_train)
                y_pred = model.predict(X_test)
                par ="Tuned Gaussian model parameters: {}".format(model.best_params_)
                best_score.update({par: model.best_score_})
                cm = confusion_matrix(y_test, y_pred)
                print(par)
                print('Best score: ' + str(model.best_score_*100) + ' %')
                print('Gaussian confusion matrix:')
                print(cm)
                if save_best:
                    export_trained_model(model, 'gaus_dtc')

    if 'lda' in hparameters:
                tree = LinearDiscriminantAnalysis()
                if search == 'random':
                    model = RandomizedSearchCV(tree, hparameters['lda'], cv=folds)
                elif search == 'grid':
                    model = GridSearchCV(tree, hparameters['lda'], cv=folds)
                model.fit(X_train,y_train)
                y_pred = model.predict(X_test)
                par = "Tuned LDA model parameters: {}".format(model.best_params_)
                best_score.update({par: model.best_score_})
                cm = confusion_matrix(y_test, y_pred)
                print(par)
                print('Best score: ' + str(model.best_score_*100) + ' %')
                print('LDA confusion matrix:')
                print(cm)
                if save_best:
                    export_trained_model(model, 'lda_dtc')

    if 'logreg' in hparameters:
                tree = LogisticRegression()
                if search == 'random':
                    model = RandomizedSearchCV(tree, hparameters['logreg'], cv=folds)
                elif search == 'grid':
                    model = GridSearchCV(tree, hparameters['logreg'], cv=folds)
                model.fit(X_train,y_train)
                y_pred = model.predict(X_test)
                par = "Tuned LogReg model parameters: {}".format(model.best_params_)
                best_score.update({par: model.best_score_})
                cm = confusion_matrix(y_test, y_pred)
                print(par)
                print('Best score: ' + str(model.best_score_*100) + ' %')
                print('Logreg confusion matrix:')
                print(cm)
                if save_best:
                    export_trained_model(model, 'logreg_dtc')
    print('\n###### Best test score: ######')
    print(str(max(best_score.items(), key=lambda k: k[1])).replace('()','')+ '\n')    
    return best_score


def run_multiple(path='', algos=[], sample_count=None, log=False, exportBest=False):
    print('\n########## Multi-algorithm testing started ##########\n')
    train_score = {}
    test_score = {}
    for i in range(len(algos)):
        training_accuracy, test_accuracy = testAlgo(path=path, algo=algos[i], samples=sample_count, export=False, log=False)
        train_score.update({algos[i]: training_accuracy})
        test_score.update({algos[i]: test_accuracy})
        print('########## ' + algos[i] + ' ##########')
        print("Train set score: {0} %".format(training_accuracy*100))
        print("Test set score: {0} %".format(test_accuracy*100)+ "\n")
    print('\nBest training score: ' + str(max(train_score.items(), key=lambda k: k[1])))
    print('Best test score: '+ str(max(test_score.items(), key=lambda k: k[1])))

def plot(path, sample_size=None, target=[], id='', kind=''):
      df = pd.read_csv(path,nrows=sample_size)
      df.groupby(target)[id].size().unstack().plot(kind=kind,stacked=True)
      plt.show()

def analyze(path, sample_count=None, save=False):
       df= pd.read_csv(path, sep = ",", nrows=sample_count, low_memory=False,  error_bad_lines=False)
       print('Starting analysis . . .\n')
       print('Dataframe has shape: ' + str(df.shape))

       print('\nIdentifing bad features:\n')

       X= df.drop(df.columns[-1], axis='columns')
       y= df[df.columns[-1]]
       fs = FeatureSelector(data = X, labels = y)
       fs.identify_all(selection_params = {'missing_threshold': 0.6, 'correlation_threshold': 0.98, 
                                    'task': 'regression', 'eval_metric': 'auc', 
                                     'cumulative_importance': 0.99})

       fs.plot_feature_importances(threshold = 0.99, plot_n = 15)

       if save:
               print('\nRemoving all bad features...')
               df = fs.remove(methods = 'all', keep_one_hot = False)
               print('Clean Dataframe has shape: ' + str(df.shape))
               print(df.head)
               df.to_csv('./clean_data/fe_clean_data_to_train.csv')
          
       return None

def encodelabels(path, row_count, log=False):
    df= pd.read_csv(path, sep = ",", nrows=row_count, low_memory=False)

    inputs = df.drop(df.columns[-1], axis='columns')
    target = df[df.columns[-1]]
    feature_list = list(inputs.columns)
    #feature_list = [float(k.strip('[] ')) for k in feature_list.split(',')]   
    for i in range(len(feature_list)):
        names = LabelEncoder()
        inputs[feature_list[i] + '_n'] = names.fit_transform(inputs[feature_list[i]])

    inputs_n = inputs.drop(feature_list, axis='columns')
    if log:
        print('Features head:' + inputs_n.head())
        print('Target ' + target.head())

    return inputs_n,target

def encodetest(path, row_count, log=False):
    df= pd.read_csv(path, sep = ",", nrows=row_count, low_memory=False)
    feature_list = list(inputs.columns)
    #feature_list = [float(k.strip('[] ')) for k in feature_list.split(',')]   
    for i in range(len(feature_list)):
        names = LabelEncoder()
        inputs[feature_list[i] + '_n'] = names.fit_transform(inputs[feature_list[i]])

    inputs_n = inputs.drop(feature_list, axis='columns')
    if log:
        print('Features head:' + inputs_n.head())
    return inputs_n

def export_trained_model(model, name):
    path = export_path + name + '.pkl'
    saved_model = pk.dump(model, open(path, 'wb'))
    print('\nModel ' + name + ' saved at ' + path + '\n')

def predict_on_model(path, feature):
    # load the model from disk and predict
    loaded_model = pk.load(open(path, 'rb'))
    result = loaded_model.predict([feature])
    print("Input = %s, Predicted = %s" % (feature, result[0]))

def extractTrainData(path, savepath, columns=[], row_count=None):
      print('\n########## ML toolbox starting ##########\n')
      print('Extracting training data...\n')
      
      if savepath == '' or len(columns) <1 or path == '':
            print('You need to specify all required parameters!')
            sys.exit(0)
      
      train_list = []
      for i in range (len(columns)):
            train_list.append(pd.read_csv(path, sep = ",", header=0, nrows=row_count, low_memory=False,  error_bad_lines=False) [columns[i]])

      export_train_data = izip_longest(*train_list, fillvalue = '')
      with open(savepath, 'w') as myfile:
                  wr = csv.writer(myfile)
                  wr.writerow(columns)
                  wr.writerows(export_train_data)
      myfile.close()
      print('\nRaw train data extracted and cleaned at ' + savepath + '\n')

def extractTestData(path, savepath, columns=[], row_count=None):
      print('\nExtracting testing data . . .')

      if savepath == '' or len(columns) < 1 or path == '':
            print('You need to specify all required parameters!')
            sys.exit(0)
      

      test_list=[]
      for i in range (len(columns)):   
            test_list.append(pd.read_csv(path, sep = ",", dtype=str,header=0, nrows=row_count, low_memory=False) [columns[i]])
      
      export_test_data = izip_longest(*test_list, fillvalue = '')
      with open(savepath, 'w') as myfile:
            wr = csv.writer(myfile)
            wr.writerow(columns)
            wr.writerows(export_test_data)
      myfile.close()
      print('Raw test data extracted and cleaned at ' + savepath)

# Algorithms

# def autoregressione_classifier(train_dataset, logs=False, export_model=False):
#     print('Training started . . .')
#     from statsmodels.tsa.ar_model import AR
    
#     # shuffle our data use 80:20 for train:test
#     random.shuffle(train_dataset)
#     n = int(len(train_dataset)*.80)

#     # create train and test cases
#     trainData = train_dataset[:n]
#     testData = train_dataset[n:]
#     # fit model
#     model = AR(trainData)
#     model_fit = model.fit()
#     # make prediction
#     train_prediction = model_fit.predict(len(trainData), len(trainData))
#     test_prediction = model_fit.predict(len(testData), len(testData))
#     print('Auto-regression results: ' + prediction)
#     if export_model:
#         export_model(model_fit, 'autoregressione_classifier')

# def movingaveragee_classifier(train_dataset, logs=False, export_model=False):
#     print('Training started . . .')
#     from statsmodels.tsa.arima_model import ARMA
#     from random import random
#     # fit model
#     model = ARMA(train_dataset, order=(0, 1))
#     model_fit = model.fit(disp=False)
#     # make prediction
#     prediction = model_fit.predict(len(train_dataset), len(train_dataset))
#     print('Moving-average results: ' + prediction)
#     if export_model:
#         export_model(model_fit, 'movingaveragee_classifier')

# def ARmovingaveragee_classifier(train_dataset, logs=False, export_model=False):
#     print('Training started . . .')
#     from statsmodels.tsa.arima_model import ARMA
#     from random import random
#     # fit model
#     model = ARMA(data, order=(2, 1))
#     model_fit = model.fit(disp=False)
#     # make prediction
#     prediction = model_fit.predict(len(train_dataset), len(train_dataset))
#     print('Autoregressive Moving Average (ARMA) results: ' + prediction)
#     if export_model:
#         export_model(model_fit, 'ARmovingaveragee_classifier')

# def ARImovingaveragee_classifier(train_dataset, logs=False, export_model=False):
#     print('Training started . . .')
#     from statsmodels.tsa.arima_model import ARIMA
#     from random import random
#     # fit model
#     model = ARIMA(data, order=(1, 1, 1))
#     model_fit = model.fit(disp=False)
#     # make prediction
#     prediction = model_fit.predict(len(train_dataset), len(train_dataset), typ='levels')
#     print('Autoregressive Moving Integrated Average (ARMA) results: ' + prediction)
#     if export_model:
#         export_model(model_fit, 'ARImovingaveragee_classifier')

# def sarima_classifier(train_dataset, logs=False, export_model=False):
#     print('Training started . . .')
#     from statsmodels.tsa.statespace.sarimax import SARIMAX
#     from random import random
#     # fit model
#     model = SARIMAX(train_dataset, order=(1, 1, 1), seasonal_order=(1, 1, 1, 1))
#     model_fit = model.fit(disp=False)
#     # make prediction
#     prediction = model_fit.predict(len(train_dataset), len(train_dataset))
#     print('Seasonal Autoregressive Integrated Moving-Average (SARIMA) results: ' + prediction)
#     if export_model:
#         export_model(model_fit, 'sarima_classifier')

# def sarimax_classifier(train_dataset, logs=False, export_model=False):
#     print('Training started . . .')
#     from statsmodels.tsa.statespace.sarimax import SARIMAX
#     from random import random
#     # contrived dataset
#     train_dataset = [x + random() for x in range(1, 100)]
#     data2 = [x + random() for x in range(101, 200)]
#     # fit model
#     model = SARIMAX(train_dataset, exog=data2, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
#     model_fit = model.fit(disp=False)
#     # make prediction
#     exog2 = [200 + random()]
#     # make prediction
#     prediction = model_fit.predict(len(train_dataset), len(train_dataset),exog=[exog2])
#     print('Seasonal Autoregressive Integrated Moving-Average with Exogenous Regressors(SARIMAX) results: ' + prediction)
#     if export_model:
#         export_model(model_fit, 'sarimax_classifier')

# def vector_autoregressione_classifier(train_dataset, logs=False, export_model=False):
#     print('Training started . . .')
#     from statsmodels.tsa.vector_ar.var_model import VAR
#     from random import random
#     # contrived dataset with dependency
#     train_dataset = list()
#     for i in range(100):
#         v1 = i + random()
#         v2 = v1 + random()
#         row = [v1, v2]
#         train_dataset.append(row)
#     # fit model
#     model = VAR(train_dataset)
#     model_fit = model.fit()
#     # make prediction
#     prediction = model_fit.forecast(model_fit.y, steps=1)
#     print('Vector Autoregression (VAR) results: ' + prediction)
#     if export_model:
#         export_model(model_fit, 'vector_autoregressione_classifier')

# def vector_autoregression_movingavr_classifier(train_dataset, logs=False, export_model=False):
#     print('Training started . . .')
#     from statsmodels.tsa.statespace.varmax import VARMAX
#     from random import random
#     # contrived dataset with dependency
#     train_dataset = list()
#     for i in range(100):
#         v1 = random()
#         v2 = v1 + random()
#         row = [v1, v2]
#         train_dataset.append(row)
#     # fit model
#     model = VARMAX(train_dataset, order=(1, 1))
#     model_fit = model.fit(disp=False)
#     # make prediction
#     prediction = model_fit.forecast()
#     print('Vector Autoregression Moving-Average (VARMA) results: ' + prediction)
#     if export_model:
#         export_model(model_fit, 'vector_autoregression_movingavr_classifier')

# def varmaxe_classifier(train_dataset, logs=False, export_model=False):
#     print('Training started . . .')
#     from statsmodels.tsa.statespace.varmax import VARMAX
#     from random import random
#     # contrived dataset with dependency
#     train_dataset = list()
#     for i in range(100):
#         v1 = random()
#         v2 = v1 + random()
#         row = [v1, v2]
#         train_dataset.append(row)
#     data_exog = [x + random() for x in range(100)]
#     # fit model
#     model = VARMAX(train_dataset, exog=data_exog, order=(1, 1))
#     model_fit = model.fit(disp=False)
#     # make prediction
#     data_exog2 = [[100]]
#     prediction = model_fit.forecast(exog=data_exog2)
#     print('Vector Autoregression Moving-Average with Exogenous Regressors (VARMAX) results: ' + prediction)
#     if export_model:
#         export_model(model_fit, 'varmaxe_classifier')

# def simple_expo_smoothinge_classifier(train_dataset, logs=False, export_model=False):
#     print('Training started . . .')
#     from statsmodels.tsa.holtwinters import SimpleExpSmoothing
#     # fit model
#     model = SimpleExpSmoothing(train_dataset)
#     model_fit = model.fit()
#     # make prediction
#     prediction = model_fit.predict(len(train_dataset), len(train_dataset))
#     print('Simple Exponential Smoothing (SES) results: ' + prediction)
#     if export_model:
#         export_model(model_fit, 'simple_expo_smoothinge_classifier')

# def holtwintere_classifier(train_dataset, logs=False, export_model=False):
#     print('Training started . . .')
#     from statsmodels.tsa.holtwinters import ExponentialSmoothing
#     # fit model
#     model = ExponentialSmoothing(train_dataset)
#     model_fit = model.fit()
#     # make prediction
#     prediction = model_fit.predict(len(train_dataset), len(train_dataset))
#     print('Holt Winters Exponential Smoothing (HWES) results: ' + prediction)
#     if export_model:
#         export_model(model_fit, 'holtwintere_classifier')




