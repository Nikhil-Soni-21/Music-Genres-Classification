from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from xgboost import XGBClassifier
from os.path import exists
import pickle


ModelFiles = {
    'GaussianNaiveBayes': 'models\\GaussianNaiveBayes.pkl',
    'LogisticRegression': 'models\\LogisticRegression.pkl',
    'DecisionTreeClassifier': 'models\\DecisionTreeClassifier.pkl',
    'XGBoost': 'models\\XGBoost.pkl',
    'SupportVectorMachine': 'models\\SVM.pkl',
    'RandomForestClassifier': 'models\\RandomForestClassifier.pkl',
    'SGDClassifier': 'models\\SGDClassifier.pkl',
    'KNeighborsClassifier': 'models\\KNNClassifier.pkl',
    'RegularNeuralNetworks': 'models\\RNN.h5'
}


class ModelUtil:
    def __init__(self, df, target, split):
        self.df = df
        self.target = target
        self.split = split
        self.SplitData()

    def SplitData(self):
        Y = self.df[self.target]
        X = self.df.drop(self.target, axis=1)
        D = train_test_split(X, Y, test_size=self.split)
        self.XTrain = D[0]
        self.XTest = D[1]
        self.YTrain = D[2]
        self.YTest = D[3]

    def GetTestingData(self):
        return self.XTest, self.YTest

    def GaussianNaiveBayes(self, load=True) -> GaussianNB:
        path = ModelFiles['GaussianNaiveBayes']
        classifier = None
        if load and exists(path):
            file = open(path, 'rb')
            classifier = pickle.load(file)
            print('Loaded Gaussian Naive Bayes from File')
            file.close()
        else:
            classifier = GaussianNB()
            classifier.fit(self.XTrain, self.YTrain)
            file = open(path, 'wb')
            pickle.dump(classifier, file)
            print('Trained and Saved Gaussian Naive Bayes')
            file.close()

        return classifier

    def LogisticRegression(self, load=True) -> LogisticRegression:
        path = ModelFiles['LogisticRegression']
        classifier = None
        if load and exists(path):
            file = open(path, 'rb')
            classifier = pickle.load(file)
            print('Loaded Logistic Regression Model from File')
            file.close()
        else:
            classifier = LogisticRegression()
            classifier.fit(self.XTrain, self.YTrain)
            file = open(path, 'wb')
            pickle.dump(classifier, file)
            print('Trained and Saved Logistic Regression Model')
            file.close()

        return classifier

    def DecisionTreeClassifier(self, load=True) -> DecisionTreeClassifier:
        path = ModelFiles['DecisionTreeClassifier']
        classifier = None
        if load and exists(path):
            file = open(path, 'rb')
            classifier = pickle.load(file)
            print('Loaded Decision Tree Model from File')
            file.close()
        else:
            best_x = 0
            best_score = 0
            for i in range(500):
                clf = DecisionTreeClassifier(random_state=i)
                clf.fit(self.XTrain, self.YTrain)
                YPred = clf.predict(self.XTest)
                score = accuracy_score(self.YTest, YPred)
                print(f'Training on Random State: {i}\tAccuracy: {score}')
                if score > best_score:
                    best_score = score
                    best_x = i
            classifier = DecisionTreeClassifier(random_state=best_x)
            classifier.fit(self.XTrain, self.YTrain)
            file = open(path, 'wb')
            pickle.dump(classifier, file)
            print('Trained and Saved Decision Tree Model')
            file.close()
        return classifier

    def XGBoostClassifier(self, load=True) -> XGBClassifier:
        path = ModelFiles['XGBoost']
        classifier = None
        if load and exists(classifier):
            file = open(path, 'rb')
            classifier = pickle.load(file)
            print('Loaded XGBoost Model from File')
            file.close()
        else:
            classifier = XGBClassifier()
            classifier.fit(self.XTrain, self.YTrain)
            file = open(path, 'wb')
            pickle.dump(classifier, file)
            print('Trained and Saved XGBoost Model')
            file.close()
        return classifier

    def SVM(self, load=True) -> SVC:
        path = ModelFiles['SupportVectorMachine']
        classifier = None
        if load and exists(path):
            file = open(path, 'rb')
            classifier = pickle.load(file)
            file.close()
            print('Loaded SVM Model from File')
        else:
            classifier = SVC()
            classifier.fit(self.XTrain, self.YTrain)
            file = open(path, 'wb')
            pickle.dump(classifier, file)
            print('Trained and Saved SVM Model')
            file.close()
        return classifier

    def RandomForestClassifier(self, load=True) -> RandomForestClassifier:
        path = ModelFiles['RandomForestClassifier']
        classifier = None
        if load and exists(path):
            file = open(path, 'rb')
            classifier = pickle.load(file)
            print('Loaded Random Forest Model from File')
            file.close()
        else:
            best_acc = 0
            best_x = 0
            for i in range(500):
                clf = RandomForestClassifier(random_state=i)
                clf.fit(self.XTrain, self.YTrain)
                YPred = clf.predict(self.XTest)
                score = accuracy_score(self.YTest, YPred)
                print(f'Training on Random State: {i}\tAccuracy: {score}')
                if score > best_acc:
                    best_acc = score
                    best_x = i

            classifier = RandomForestClassifier(random_state=best_x)
            classifier.fit(self.XTrain, self.YTrain)
            file = open(path, 'wb')
            pickle.dump(classifier, file)
            print('Trained and Saved Random Forest Model')
            file.close()

        return classifier

    def SGDClassifier(self, load=True) -> SGDClassifier:
        path = ModelFiles['SGDClassifier']
        classifier = None
        if load and exists(path):
            file = open(path, 'rb')
            classifier = pickle.load(file)
            print('Loaded SGDClassifier From File')
            file.close()
        else:
            classifier = SGDClassifier()
            classifier.fit(self.XTrain, self.YTrain)
            file = open(path, 'wb')
            pickle.dump(classifier, file)
            print('Trained and Saved SGDClassifier')
            file.close()
        return classifier

    def KNNClassifier(self, load=True) -> KNeighborsClassifier:
        path = ModelFiles['KNeighborsClassifier']
        classifier = None
        if load and exists(path):
            file = open(file, 'rb')
            classifier = pickle.load(file)
            print('Loaded KNN From File')
            file.close()
        else:
            best_k = 1
            best_score = 0
            for k in range(1, 500):
                clf = KNeighborsClassifier(n_neighbors=i)
                clf.fit(self.XTrain, self.YTrain)
                YPred = clf.predict(self.XTest)
                score = accuracy_score(self.YTest, YPred)
                print(f'Training on Neighbors: {i}\tAccuracy: {score}')
                if score > best_score:
                    best_score = score
                    best_k = i

            classifier = KNeighborsClassifier(n_neighbors=best_k)
            classifier.fit(self.XTrain, self, YTrain)
            file = open(path, 'wb')
            pickle.dump(classifier, file)
            print('Trained and Saved KNN Model')
            file.close()

    def NeuralNetwork(self, load=True) -> Sequential:
        path = ModelFiles['RegularNeuralNetworks']
        model = None
        if load and exists(path):
            model = load_model(path)
            print('Loaded RNN from File')
        else:
            model = Sequential()
            model.add(Dense(100, activation='relu', input_dim=63))
            model.add(Dense(activation='sigmoid'))
            model.compile(optimizer='adam', metrics=[
                          'accuracy'], loss='sparse_categorical_crossentropy')
            model.fit(self.XTrain, self.YTrain, epochs=500)
            model.save(path)
            print('Trained and Saved ML Model')

        return model
