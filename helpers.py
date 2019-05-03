# data analysis and wrangling
import pandas as pd
import numpy as np
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

class Helpers:
    def add_title(self, train_df, test_df, original_train_df, original_test_df, combine, replacements, title_mapping):
        # Haal alle onderdelen eruit
        train_df['Title'] = original_train_df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
        test_df['Title'] = original_train_df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

        # Vervang te specifieke namen door meer algemene namen
        for dataset in combine:
            for key, value in replacements.items():
                dataset['Title'] = dataset['Title'].replace(key, value)

        for dataset in combine:
            dataset['Title'] = dataset['Title'].map(title_mapping)
            dataset['Title'] = dataset['Title'].fillna(0)
        
        print(train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean().sort_values(by='Survived', ascending=False))
        
        return train_df, test_df, combine

    def add_age_binned(self, train_df, test_df, original_train_df, original_test_df, combine):
        guess_ages = np.zeros((2,3))
    
        train_df['Sex'] = original_train_df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
        test_df['Sex'] = original_test_df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
        train_df['Age'] = original_train_df['Age']
        test_df['Age'] = original_test_df['Age']

        for dataset in [train_df, test_df]:
            for i in range(0, 2):
                for j in range(0, 3):
                    guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j+1)]['Age'].dropna()
                    age_guess = guess_df.median()
                    guess_ages[i,j] = int(age_guess/0.5 + 0.5) * 0.5 # Convert random age float to nearest .5 age
            for i in range(0, 2):
                for j in range(0, 3):
                    dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1), 'Age'] = guess_ages[i,j]
            dataset['Age'] = dataset['Age'].astype(int)

        print(">> schattingen")
        print(train_df.head())

        # TODO: aantal bins als parameter meegeven
        train_df['AgeBand'] = pd.cut(original_train_df['Age'], 5)

        print()
        print(">> Leeftijds-bin")
        print(train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True))

        for dataset in combine:    
            dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
            dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
            dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
            dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
            dataset.loc[ dataset['Age'] > 64, 'Age']

        train_df = train_df.drop(['AgeBand'], axis=1)
        combine = [train_df, test_df]
        
        return train_df, test_df, combine
    
    def add_age_times_class(self, train_df, test_df, original_train_df, original_test_df, combine):
        train_df['Age'] = original_train_df['Age']
        train_df.loc[train_df.Age.isnull(), 'Age'] = 0

        test_df['Age'] = original_test_df['Age']
        test_df.loc[test_df.Age.isnull(), 'Age'] = 0

        train_df['Age*Class'] = train_df.Age * original_train_df.Pclass
        test_df['Age*Class'] = test_df.Age * original_train_df.Pclass

        train_df = train_df.drop("Age", axis=1)
        test_df = test_df.drop("Age", axis=1)
        
        return train_df, test_df, combine
    
    def add_family_size(self, train_df, test_df, original_train_df, original_test_df, combine):
        train_df['SibSp'] = original_train_df['SibSp']
        test_df['SibSp'] = original_test_df['SibSp']
        train_df['Parch'] = original_train_df['Parch']
        test_df['Parch'] = original_test_df['Parch']

        for dataset in combine:
            dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
        print(train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False))

        train_df = train_df.drop(['Parch', 'SibSp'], axis=1)
        test_df = test_df.drop(['Parch', 'SibSp'], axis=1)
        combine = [train_df, test_df]
        
        return train_df, test_df, combine
    
    def add_is_alone(self, train_df, test_df, original_train_df, original_test_df, combine):
        train_df['SibSp'] = original_train_df['SibSp']
        test_df['SibSp'] = original_test_df['SibSp']
        train_df['Parch'] = original_train_df['Parch']
        test_df['Parch'] = original_test_df['Parch']

        for dataset in combine:
            dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
            dataset['IsAlone'] = 0
            dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

        print(train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())

        train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
        test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
        combine = [train_df, test_df]
        
        return train_df, test_df, combine
    
    def prepare_dataset(self):
        train_df = pd.read_csv('train.csv')
        test_df = pd.read_csv('test.csv')
        
        train_df = train_df.drop(['Ticket', 'Cabin', 'Name', 'Age', 'PassengerId', 'Parch', 'SibSp'], axis=1)
        test_df = test_df.drop(['Ticket', 'Cabin', 'Name', 'Age', 'Parch', 'SibSp'], axis=1)
        
        freq_port = train_df.Embarked.dropna().mode()[0]
        test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
        for dataset in [train_df, test_df]:
            if is_string_dtype(dataset['Sex']):
                dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
            if is_string_dtype(dataset['Embarked']):
                dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
                dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
            if dataset['Fare'].dtype == np.float64:
                dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
                dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
                dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
                dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
                dataset['Fare'] = dataset['Fare'].astype(int)
        
        combine = [train_df, test_df]
        
        train_orig_df = pd.read_csv('train.csv')
        test_orig_df = pd.read_csv('test.csv')
        
        original_combine = [train_orig_df, test_orig_df]
        
        return train_df, test_df, combine, original_combine, train_orig_df, test_orig_df
    
    def clear_features(self, train_df, test_df, combine, feature_name):
        if feature_name in train_df.columns:
            train_df = train_df.drop([feature_name], axis=1)
        if feature_name in test_df.columns:
            test_df = test_df.drop([feature_name], axis=1)
        combine = [train_df, test_df]
        
        return train_df, test_df, combine
        
    def apply_naivebayes(self, X_train, Y_train, X_test):
        gaussian = GaussianNB()
        gaussian.fit(X_train, Y_train)
        Y_pred = gaussian.predict(X_test)
        acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
        print("Naive bayes score: " + str(acc_gaussian))
        return acc_gaussian
    
    def apply_logit(self, X_train, Y_train, X_test):
        logreg = LogisticRegression()
        logreg.fit(X_train, Y_train)
        Y_pred = logreg.predict(X_test)
        acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
        print("Logit score: " + str(acc_log))
        return acc_log
        
    def apply_svm(self, X_train, Y_train, X_test):
        svm = SVC()
        svm.fit(X_train, Y_train)
        Y_pred = svm.predict(X_test)
        acc_svm = round(svm.score(X_train, Y_train) * 100, 2)
        print("SVM score: " + str(acc_svm))
        return acc_svm
        
    def apply_knn(self, X_train, Y_train, X_test):
        knn = KNeighborsClassifier(n_neighbors = 3)
        knn.fit(X_train, Y_train)
        Y_pred = knn.predict(X_test)
        acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
        print("KNN score: " + str(acc_knn))
        return acc_knn
        
    def apply_perceptron(self, X_train, Y_train, X_test):
        perceptron = Perceptron(max_iter=5, tol=None)
        perceptron.fit(X_train, Y_train)
        Y_pred = perceptron.predict(X_test)
        acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
        print("Perceptron score: " + str(acc_perceptron))
        return acc_perceptron
        
    def apply_linear_svc(self, X_train, Y_train, X_test):
        linear_svc = LinearSVC()
        linear_svc.fit(X_train, Y_train)
        Y_pred = linear_svc.predict(X_test)
        acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
        print("Linear SVC score: " + str(acc_linear_svc))
        return acc_linear_svc
        
    def apply_sgd(self, X_train, Y_train, X_test):
        sgd = SGDClassifier(max_iter=5, tol=None)
        sgd.fit(X_train, Y_train)
        Y_pred = sgd.predict(X_test)
        acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
        print("SGD score: " + str(acc_sgd))
        return acc_sgd
    
    def apply_decisiontree(self, X_train, Y_train, X_test):
        decision_tree = DecisionTreeClassifier()
        decision_tree.fit(X_train, Y_train)
        Y_pred = decision_tree.predict(X_test)
        acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
        print("Decision tree score: " + str(acc_decision_tree))
        return acc_decision_tree
    
    def apply_randomforest(self, X_train, Y_train, X_test, n_estimators=100):
        random_forest = RandomForestClassifier(n_estimators)
        random_forest.fit(X_train, Y_train)
        Y_pred = random_forest.predict(X_test)
        random_forest.score(X_train, Y_train)
        acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
        print("Random forest score: " + str(acc_random_forest))
        return acc_random_forest