import numpy as np
from sklearn import preprocessing, svm, neighbors, model_selection, linear_model, tree, neural_network, ensemble
import pandas as pd
import matplotlib.pyplot as plt


def k_nearest_neighbors(X_train, X_test, y_train, y_test):
    training_accuracy = []
    test_accuracy = []
    classifier = neighbors.KNeighborsClassifier(n_neighbors=8)
    classifier.fit(X_train, y_train)
    train_accuracy = classifier.score(X_train, y_train)
    testing_accuracy = classifier.score(X_test, y_test)
    training_accuracy.append(train_accuracy)
    test_accuracy.append(testing_accuracy)
    print("Training_Accuracy:%f Test_Accuracy:%f" % (train_accuracy, testing_accuracy))
    plt.plot(range(1, 11), training_accuracy, label = 'Training accuracy')
    plt.plot(range(1, 11), test_accuracy, label='Test accuracy')
    plt.xlabel('Value of K')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    return


def logistic_regression_classification(X_train, X_test, y_train, y_test):
    classifier = linear_model.LogisticRegression(C=1)
    classifier1 = linear_model.LogisticRegression(C=0.01)
    classifier2 = linear_model.LogisticRegression(C=100)
    classifier.fit(X_train, y_train)
    classifier1.fit(X_train, y_train)
    classifier2.fit(X_train, y_train)
    train_accuracy = classifier.score(X_train, y_train)
    test_accuracy = classifier.score(X_test, y_test)
    print("Training_Accuracy:%f Test_Accuracy:%f" % (train_accuracy, test_accuracy))
    train_accuracy = classifier1.score(X_train, y_train)
    test_accuracy = classifier1.score(X_test, y_test)
    print("Training_Accuracy:%f Test_Accuracy:%f" % (train_accuracy, test_accuracy))
    train_accuracy = classifier2.score(X_train, y_train)
    test_accuracy = classifier2.score(X_test, y_test)
    print("Training_Accuracy:%f Test_Accuracy:%f" % (train_accuracy, test_accuracy))
    return


def decision_tree_classification(X_train, X_test, y_train, y_test):
    training_accuracy = []
    testing_accuracy = []
    for i in range(1, 11):
        classifier = tree.DecisionTreeClassifier(max_depth=i)
        classifier.fit(X_train, y_train)
        train_accuracy = classifier.score(X_train, y_train)
        test_accuracy = classifier.score(X_test, y_test)
        training_accuracy.append(train_accuracy)
        testing_accuracy.append(test_accuracy)
        print("Training_Accuracy:%f Test_Accuracy:%f" % (train_accuracy, test_accuracy))
        print("Feature importance:", classifier.feature_importances_)
    plt.plot(range(1, 11), training_accuracy, label='Training accuracy')
    plt.plot(range(1, 11), testing_accuracy, label='Test accuracy')
    plt.xlabel('Depth of decision tree')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    return


def support_vector_machine(X_train, X_test, y_train, y_test):
    classifier = svm.SVC()
    classifier.fit(X_train, y_train)
    train_accuracy = classifier.score(X_train, y_train)
    test_accuracy = classifier.score(X_test, y_test)
    print("Training_Accuracy:%f Test_Accuracy:%f" % (train_accuracy, test_accuracy))
    return


def random_forest_classification(X_train, X_test, y_train, y_test):
    classifier = ensemble.RandomForestClassifier(max_depth=4, n_estimators=100)
    classifier.fit(X_train, y_train)
    train_accuracy = classifier.score(X_train, y_train)
    test_accuracy = classifier.score(X_test, y_test)
    print("Training_Accuracy:%f Test_Accuracy:%f" % (train_accuracy, test_accuracy))
    return


df = pd.read_csv("diabetes.csv")
X = np.array(df.drop('Outcome', 1))
y = np.array(df['Outcome'])
# getting all features in range of [-1, 1]
X = preprocessing.scale(X)
# dividing data into training and testing ratio:80%-20%
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)


print('K - Nearest Neighbors:', end='')
k_nearest_neighbors(X_train, X_test, y_train, y_test)
print('Logistic Regression:', end='')
logistic_regression_classification(X_train, X_test, y_train, y_test)
print('Decision Tree:', end='')
decision_tree_classification(X_train, X_test, y_train, y_test)
print('Support Vector Machine:', end='')
support_vector_machine(X_train, X_test, y_train, y_test)
print('Random Forest:', end='')
random_forest_classification(X_train, X_test, y_train, y_test)
