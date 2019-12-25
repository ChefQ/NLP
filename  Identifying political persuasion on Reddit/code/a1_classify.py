from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from scipy import stats
import numpy as np
import argparse
import csv
from sklearn.utils import shuffle
import sys
import os

def accuracy( C ):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    # sum of diagonal
    correctPredictions = np.trace(C)
    predictions = np.sum(C)
    return float(correctPredictions)/predictions

def recall( C ):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    recalls = []
    for i in range(4):
        row = C[i, :]
        rec =  C[i, i] / row.sum()
        recalls.append(rec)
    return recalls

def precision( C ):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    precisions = []
    for i in range(4):
        col = C[:, i]
        preci = C[i, i] / col.sum()
        precisions.append(preci)
    return precisions


def class31(filename):
    ''' This function performs experiment 3.1
    
    Parameters
       filename : string, the name of the npz file from Task 2

    Returns:      
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier
    '''


    # NOTE this need to be changed, because the arguement is supposed to be a string (filename)

    data = np.load(filename)
    key = data.files
    feats = data[key[0]] # NOTE not sure i gotta do key[0]

    # split data
    X = feats[ :,0:173]
    Y = feats[:,173]

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2)

    # then use classifiers
    # SVC 1
    svc1 = SVC(max_iter= 1000)
    svc1.fit(Xtrain, Ytrain)

    # SVC 2
    svc2 = SVC(gamma='auto', max_iter= 1000)
    svc2.fit(Xtrain, Ytrain)

    # RandomForestClassifier
    rfc = RandomForestClassifier(n_estimators=10, max_depth=5)
    rfc.fit(Xtrain,Ytrain)

    # MLPClassifier
    mlp = MLPClassifier(alpha=0.05)
    mlp.fit(Xtrain, Ytrain)

    # AdaBoostClassifier
    ada =  AdaBoostClassifier()
    ada.fit(Xtrain, Ytrain)


    # Make prediction
    # 1
    preSvc1 = svc1.predict(Xtest)
    #2
    preSvc2 = svc2.predict(Xtest)
    #3
    preRfc1 = rfc.predict(Xtest)
    #4
    preMlp = mlp.predict(Xtest)
    #5
    preAda = ada.predict(Xtest)

    # Obtain confusion Matrix
    predicts = np.array([preSvc1,preSvc2,preRfc1,preMlp, preAda])
    confusions = []
    for label in range(5):
        predict = predicts[label]
        cm = confusion_matrix(Ytest, predict)
        confusions.append(cm)

    #calculate Accuracy, Recall, and Precison
    accuracys = []
    recalls = []
    precisions = []
    for label in range(5):
        #Accuracy
        accu = accuracy(confusions[label])
        accuracys.append(accu)
        #Recall
        rec = recall(confusions[label])
        recalls.append(rec)
        #Precision
        prec = precision(confusions[label])
        precisions.append(prec)
    #write stuff to a file

    #ask if the format you have is gucci: it is not
    with open('a1_3.1.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        for label in range(5):

            row1 = confusions[label][0, :].tolist()
            row2 = confusions[label][1, :].tolist()
            row3 = confusions[label][2, :].tolist()
            row4 = confusions[label][3, :].tolist()
            accu_str = accuracys[label]
            recall_str = recalls[label]
            prec_str = precisions[label]
            result = [label]+[accu_str] + recall_str + prec_str + row1 + row2 + row3 + row4
            writer.writerow(result)
    csvFile.close()

    iBest = np.argmax(accuracys)

    return Xtrain, Xtest, Ytrain, Ytest, iBest

def class32(X_train, X_test, y_train, y_test,iBest): #done recheck tho
    ''' This function performs experiment 3.2
    
    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   '''


    #find best classifier
    classifiers = [

                SVC(max_iter= 1000),                                    #0
                SVC(gamma='auto'),                                      #1
                RandomForestClassifier(n_estimators=10, max_depth=5),   #2
                MLPClassifier(alpha=0.05),                              #3
                AdaBoostClassifier()                                    #4

                  ]

    best = classifiers[iBest]

    # Question they said we can sample this arbitrarily, i sample in order
    # sample 1K, 5K, 10K, 15K, and 20K

    Xbatch1, Ybatch1 = X_train[0:1000, :], y_train[0:1000]
    Xbatch2, Ybatch2 = X_train[1000:6000, :], y_train[1000:6000]
    Xbatch3, Ybatch3 = X_train[6000:16000, :], y_train[6000:16000]
    Xbatch4, Ybatch4 = X_train[16000:21000, :], y_train[16000: 21000]
    Xbatch5, Ybatch5 = X_train[12000:32000, :], y_train[12000:32000]

    batches = [(Xbatch1, Ybatch1), (Xbatch2, Ybatch2), (Xbatch3, Ybatch3), (Xbatch4, Ybatch4), (Xbatch5, Ybatch5)]

    accuracys = []
    for batch in batches:
        # train each using only the best
        best.fit(batch[0] ,batch[1])

        # Make prediction
        predict = best.predict(X_test)

        # obtain confusion matrix
        cm = confusion_matrix(y_test, predict)

        # calculate accuracy
        accu = accuracy(cm)
        accuracys.append(accu)

    #write the accuracies in the csv file
    with open('a1_3.2.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(accuracys)
    csvFile.close()

    return Xbatch1, Ybatch1
    
def class33(X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3
    
    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''
    ks = [5, 10, 20, 30, 40, 50]

    # find features for 1k

    p_values1k = []
    for p in ks:
        selector = SelectKBest(f_classif, k=p)
        if (p == 5):
            Xnew_1k = selector.fit_transform(X_1k, y_1k)
        else:
            selector.fit_transform(X_1k, y_1k)
        indices = selector.get_support(indices = True)
        pp = selector.pvalues_[indices]
        p_values1k.append(pp)

    # find features for 32K

    p_values32k = []
    for p in ks:
        selector = SelectKBest(f_classif, k=p)
        if (p == 5):
            Xnew_32k = selector.fit_transform(X_train, y_train)
        else:
            selector.fit_transform(X_1k, y_1k)
        indices = selector.get_support(indices = True)
        pp = selector.pvalues_[indices]
        p_values32k.append(pp)



    # write pvalues for 32K to csv
    with open('a1_3.3.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        # number of features and assocciated pvalues
        count = 0
        for k in ks:
            p_values = p_values32k[count]
            result = [k] + p_values.tolist()
            writer.writerow(result)
            count += 1
    csvFile.close()

    # find best classifier
    classifiers = [

        SVC(max_iter=1000),  # 0
        SVC(gamma='auto', max_iter=1000),  # 1
        RandomForestClassifier(n_estimators=10, max_depth=5),  # 2
        MLPClassifier(alpha=0.05),  # 3
        AdaBoostClassifier()  # 4

    ]

    best = classifiers[i]

    #select the right features for Xtest #NOTE i'm not sure if this is what i'm suppossed to do

    selector = SelectKBest(f_classif, k=5)
    Xnew_test = selector.fit_transform(X_test, y_test)

    # train best classifier for 1k
    best.fit(Xnew_1k, y_1k)
    predict = best.predict(Xnew_test)
    cm = confusion_matrix(y_test, predict)
    accu1k = accuracy(cm)

    # train best classifier for 32k

    best.fit(Xnew_32k, y_train)
    predict = best.predict(Xnew_test)
    cm = confusion_matrix(y_test, predict)
    accu32k = accuracy(cm)


    # write to csv file
    with open('a1_3.3.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        # number of features and assocciated pvalues
        result = [accu1k, accu32k]
        writer.writerow(result)
    csvFile.close()


def class34( filename, i ): #complete
    ''' This function performs experiment 3.4
    
    Parameters
       filename : string, the name of the npz file from Task 2
       i: int, the index of the supposed best classifier (from task 3.1)  
        '''


    data = np.load(filename) # NOTE this is not how it's supposed to be done, i think
    key = data.files
    feats = data[key[0]] #NOTE not sure i gotta do key[0]



    # get date
    X = feats[:, 0:173]
    Y = feats[:, 173]


    #get classifiers
    classifiers = [

                SVC(max_iter = 1000),                                   #0
                SVC(gamma='auto', max_iter = 1000),                     #1
                RandomForestClassifier(n_estimators=10, max_depth=5),   #2
                MLPClassifier(alpha=0.05),                              #3
                AdaBoostClassifier()                                    #4

                  ]
    # split data into 5 folds
    kf = KFold(n_splits=5, random_state=None, shuffle=True)
    all_accuracys = []
    for train_index, test_index in kf.split(X):
        accuracys = []
        # for each of the classifiers run 5-fold cross-validation
        for classifier in classifiers:
            # train and test classifiers
            classifier.fit(X[train_index], Y[train_index])
            # calculate accuracy
            predict = classifier.predict(X[test_index])
            # obtain confusion matrix
            cm = confusion_matrix(Y[test_index], predict)
            # calculate accuracy
            accu = accuracy(cm)
            accuracys.append(accu)

        all_accuracys.append(accuracys)
    # write to csv file
    with open('a1_3.4.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(all_accuracys)
    csvFile.close()

    #Calculates the T-test on TWO RELATED samples of scores, a and b.
    a = all_accuracys.pop(i)
    p_values = []
    for b in all_accuracys:
        result = stats.ttest_rel(a, b)
        p_value = result.pvalue
        p_values.append(p_value)

    with open('a1_3.4.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(p_values)
    csvFile.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    args = parser.parse_args()

    # Get file hre
    # TODO : complete each classification experiment, in sequence.


    # class31
    Xtrain, Xtest, Ytrain, Ytest, iBest = class31(args.input)


    # class32
    X_1k, y_1k = class32(Xtrain, Xtest, Ytrain, Ytest, iBest)


    # class33
    class33(Xtrain, Xtest, Ytrain, Ytest, iBest, X_1k, y_1k)


    class34(args.input, iBest)
