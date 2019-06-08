import numpy as np
from sklearn import svm
from sparse import PCAResultReader, ClassReader
from PCA import GetPCAResult
from ReadClass import GetClass
import pickle
import matplotlib.pyplot as plt

class SVMClassifier:
    def __init__(self, pcareader, classreader, diminsion=50, train_proportion=0.7):
        self.classifier = svm.SVC()
        data = pcareader.data
        self.diminsion = diminsion
        label = classreader.sample_label
        self.type_number = classreader.type_number
        self.ChooseData(data=data, label=label, train_proportion=train_proportion)

    def ChooseData(self, data, label, train_proportion):
        data = np.delete (data, np.where (label == -1), axis=1).transpose()
        label = np.delete (label, np.where (label == -1), axis=0)
        rand_ = np.random.random([data.shape[0]])
        self.train_data = data[rand_ <= train_proportion, :]
        self.train_label = label[rand_ <= train_proportion]
        self.test_data = data[rand_ > train_proportion, :]
        self.test_label = label[rand_ > train_proportion]

    def fit(self):
        self.classifier.fit(self.train_data[:, :self.diminsion], self.train_label)

    def test(self):
        result = self.classifier.predict(self.test_data[:, :self.diminsion])
        precision = np.sum (result == self.test_label) / result.shape
        result_train = self.classifier.predict(self.train_data[:, :self.diminsion])
        precision_train = np.sum (result_train == self.train_label) / result_train.shape
        return precision, precision_train

    def class_anal(self):
        print ("_train dataset _____________:")
        for i in range(self.type_number.__len__()):
            samp_data = self.train_data[self.train_label == i, :]

            result = self.classifier.predict(samp_data[:, :self.diminsion])
            precision = np.sum (result == i) / result.shape
            print ('class ', i, 'totally ', samp_data.shape[0], ' samples, precision : ', precision)

        print ("_test dataset _____________:")
        for i in range(self.type_number.__len__()):
            samp_data = self.test_data[self.test_label == i, :]

            result = self.classifier.predict(samp_data[:, :self.diminsion])
            precision = np.sum (result == i) / result.shape
            print ('class ', i, 'totally ', samp_data.shape[0], ' samples, precision : ', precision)

    def pca_anal(self):
        picklefile = open('Gene_Chip_Data/Gene_Chip_Data/pca_artibutes.txt', 'rb')
        [eigenvalue, eigenvector, component_variance, singular, mean_, var_, noise_] = pickle.load(picklefile)
        #print (a)
        picklefile.close()
        plt.bar(range(len(eigenvalue)), eigenvalue)
        plt.xlabel('eigenvalue')
        plt.savefig('./figure/eigenvalue.pdf')
        plt.close('all')
        #print (eigenvalue)
        #print (eigenvector)

if __name__ == '__main__':
    pcareader = GetPCAResult()
    classreader = GetClass()
    clf = SVMClassifier (pcareader=pcareader, classreader=classreader)
    clf.fit()
    print (clf.test())
