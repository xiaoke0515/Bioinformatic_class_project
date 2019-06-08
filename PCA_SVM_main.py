from SVM import SVMClassifier
from PCA import GetPCAResult
from ReadClass import GetClass

class_reader = GetClass()
pca_reader = GetPCAResult (reduced_component=50, batch_size=100)
clf = SVMClassifier (pcareader=pca_reader, classreader=class_reader, diminsion=10)
clf.fit()
precision, precision_train = clf.test()
print ('test accuracy: ', precision, ' train accuracy: ', precision_train)
clf. class_anal()
clf.pca_anal()

