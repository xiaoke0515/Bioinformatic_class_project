from ReadClass import GetClass
from NN import NN
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    

class_reader = GetClass()
filename = "Gene_Chip_Data/Gene_Chip_Data/microarray.original.txt"
classifier = NN(class_reader, filename, train_proportion=0.7)
(sess, accuracy, loss, input_data, label) = classifier.BuildGraph(200, 50, 20)
#classifier.Train(sess, accuracy, loss, input_data, label, batch_size=100, train_time=500, lr=1e-3)
classifier.Test(sess, accuracy, loss, input_data, label, batch_size=100)
