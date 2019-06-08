import numpy as np
import csv
import pathlib

class DataReader:
#    __slots__ = ('batch_size')
    sample = []
    sample_number = 0
    gene = []
    gene_number = 0
    batch_size = 100
    batch_end = False
    batch_indicator = 0
    #__csv_file
    #__csv_reader
    __delimiter = '\t'
    def __init__(self, filename, batch_size=100, delimiter='\t'):
        self.__delimiter = delimiter
        self.batch_end = False
        self.batch_indicator = 0
        self.__csv_file = open (filename)
        self.__csv_reader = csv.reader (self.__csv_file, delimiter=self.__delimiter)
        # get sample name
        item = self.__csv_reader.__next__()
        self.sample = item[1: ]
        self.sample_number = self.sample.__len__()
        # get gene number
        print ('getting gene number')
        for row in self.__csv_reader:
            self.gene_number += 1
            self.gene.append(row[0])
            if self.__csv_reader.line_num % 1000 == 0:
                print ("\t read line ", self.__csv_reader.line_num)
        print ('totally %d genes, %d samples' % (self.gene_number, self.sample_number))
    
    def __del__ (self):
        self.__csv_file.close()
    def GetBatch(self):
        self.batch_end = False
        self.__csv_file.seek(0)
        self.__csv_reader = csv.reader (self.__csv_file, delimiter=self.__delimiter)
        n_sample = min (self.batch_size, self.sample_number - self.batch_indicator)
        data = np.zeros ([self.gene_number, self.batch_size])
        for item in self.__csv_reader:
            if self.__csv_reader.line_num < 2:
                continue
            if self.__csv_reader.line_num > self.gene_number + 1:
                break
            nums = [float(n) for n in item[1:]]
            nums_norm = (nums - np.min(nums)) / (np.max(nums) - np.min(nums))
            #print (n_sample)
            #print (nums_norm.shape)
            #print (nums_norm[self.batch_indicator: self.batch_indicator + n_sample].shape)
            data[self.__csv_reader.line_num - 2, 0: n_sample] = nums_norm[self.batch_indicator : self.batch_indicator + n_sample]
            if self.__csv_reader.line_num % 1000 == 0:
                print ("\t readline ", self.__csv_reader.line_num)
        self.batch_indicator += self.batch_size
        if self.batch_indicator >= self.sample_number:
            self.batch_end = True
            self.batch_indicator = 0
        return data

    def Reset (self):
        self.batch_indicator = 0
        self.batch_end = False

class PCAResultReader:
    sample = []
    sample_number = 0
    factor = []
    factor_number = 0
    data = np.zeros([])
    def __init__(self, filename, delimiter='\t'):
        self.__csv_file = open(filename)
        self.__csv_reader = csv.reader (self.__csv_file, delimiter=delimiter)
        #get sample name
        item = self.__csv_reader.__next__()
        self.sample = item
        self.sample_number = item.__len__()
        self.data = []
        for item in self.__csv_reader:
            self.data.append(item)
            self.factor_number += 1
        self.data = np.array(self.data)

    def __del__(self):
        self.__csv_file.close()

class ClassReader:
    class_column = -1
    sample_type = []
    sample_label = []
    label_type = [  ('normal', ['normal', 'health']),
                    ('leukemia', ['leukemia', 'AML']),
                    ('atopic', ['atopic']),
                    ('B-cell', ['B-cell']),
                    ('breast', ['breast']), 
                    ('brain', ['brain', 'Huntington']), 
                    ('bone', ['bone', 'cho']),
                    ('lung', ['lung']),
                    ('bladder', ['bladder']),
                    ('cololrectal', ['colo'])]
    type_number = []
    def __init__(self, filename, delimter='\t'):
        self.__csv_file = open(filename)
        self.__csv_reader = csv.reader (self.__csv_file, delimiter=delimter)
        # get class position
        item = self.__csv_reader.__next__()
        for index, title in enumerate(item):
            if 'DiseaseState' in title:
                self.class_column = index
                break
        self.sample_type = []
        for item in self.__csv_reader:
            self.sample_type.append (item[self.class_column])
        self.type_number = np.zeros([len(self.label_type)], dtype=np.int16)
        self.get_label (self.sample_type)

    def __del__(self):
        self.__csv_file.close()

    def get_label (self, sample_type):
        for index, item in enumerate(sample_type):
            flag = False
            for index_1, (T, kws) in enumerate(self.label_type):
                for kw in kws:
                    if kw in item:
                        self.sample_label.append (index_1)
                        flag = True
                        self.type_number[index_1] += 1
                        break
                if flag == True:
                    break
            if not flag:
                self.sample_label.append (-1)
    
    def PrintTypeNumber(self):
        print ('name, number')
        for index, (T, kews) in enumerate (self.label_type):
            print (T, ': ', self.type_number[index])
        print ('totally :', np.sum(self.type_number))

class NNDataReader:
#    __slots__ = ('batch_size')
    def __init__(self, filename, batch_size=100, delimiter='\t'):
        self.__delimiter = delimiter
        self.batch_size = batch_size
        self.batch_end_train = False
        self.batch_indicator_train = 0
        self.batch_end_test = False
        self.batch_indicator_test = 0

    def HaveResult(self):
        self.train_filename = "Gene_Chip_Data/Gene_Chip_Data/nn_train.txt"
        self.test_filename = "Gene_Chip_Data/Gene_Chip_Data/nn_test.txt"
        #self.label_filename = "Gene_Chip_Data/Gene_Chip_Data/nn_label.txt"
        if (not pathlib.Path(self.train_filename).is_file()) or (not pathlib.Path(self.test_filename).is_file()):# or (not pathlib.Path(self.label_filename).is_file()):
            return False
        else:
            return True

    def Reform (self, filename, label, train_proportion):
        used_data = np.where(label != -1)[0]
        # random choose train/test
        rand_ = np.random.random(used_data.shape)
        train_data = used_data[rand_ <= train_proportion]
        np.random.shuffle (train_data)
        train_label = label[train_data]
        test_data = used_data[rand_ > train_proportion]
        np.random.shuffle (test_data)
        test_label = label[test_data]

        # read file
        csv_file = open (filename, newline='')
        csv_reader = csv.reader (csv_file, delimiter=self.__delimiter)
        # get sample name
        item = csv_reader.__next__()
        self.sample_train = [item[1: ][i] for i in train_data]
        self.sample_test = [item[1: ][i] for i in test_data]
        self.train_number = self.sample_train.__len__()
        self.test_number = self.sample_test.__len__()
        # get gene number
        print ('getting gene number')
        self.gene_number = 0
        self.gene = []
        for row in csv_reader:
            self.gene_number += 1
            self.gene.append(row[0])
            if csv_reader.line_num % 1000 == 0:
                print ("\t read line ", csv_reader.line_num)
        print ('totally %d genes, %d train samples, %d test samples' % (self.gene_number, self.train_number, self.test_number))
        self.ReformData(csv_file, train_data, test_data, train_label, test_label)

    def GetData(self):
        #get reader
        self.__csv_file_train = open (self.train_filename)
        self.__csv_reader_train = csv.reader (self.__csv_file_train, delimiter=self.__delimiter)
        self.__csv_file_test = open (self.test_filename)
        self.__csv_reader_test = csv.reader (self.__csv_file_test, delimiter=self.__delimiter)
        # get gene number
        item = self.__csv_reader_train.__next__()
        self.gene = item[2:]
        self.gene_number = self.gene.__len__()
        # get train sample
        self.train_number = 0
        self.sample_train = []
        self.train_label = np.array([])
        for item in self.__csv_reader_train:
            if self.__csv_reader_train.line_num < 2:
                continue
            self.train_number += 1
            self.sample_train.append (item[0])
            self.train_label = np.append(self.train_label, float(item[1]))
        self.__csv_file_train.seek(0)
        self.__csv_reader_train = csv.reader (self.__csv_file_train, delimiter=self.__delimiter)
        # get test sample
        self.test_number = 0
        self.sample_test = []
        self.test_label = np.array([])
        for item in self.__csv_reader_test:
            if self.__csv_reader_test.line_num < 2:
                continue
            self.test_number += 1
            self.sample_test.append (item[0])
            self.test_label = np.append(self.test_label, float(item[1]))
        self.__csv_file_test.seek(0)
        self.__csv_reader_test = csv.reader (self.__csv_file_test, delimiter=self.__delimiter)
        # get type number
        label = np.append (self.train_label, self.test_label)
        self.type_number = np.array(list(set(label.tolist()))).shape[0]
        print ('totally ', self.train_number, ' train samples, ', self.test_number, ' test samples, ', self.gene_number, ' genes')

    
    def __del__ (self):
        self.__csv_file_train.close()
        self.__csv_file_test.close()

    def ReformData(self, csv_file, train_data, test_data, train_label, test_label):
        write_file = open (self.train_filename, 'w', newline='')
        write_csv = csv.writer (write_file, delimiter=self.__delimiter)
        write_csv.writerow (['samples', 'label'] + self.gene)
        #reform train data
        for i in range (0, self.train_number, self.batch_size):
            csv_file.seek(0)
            csv_reader = csv.reader (csv_file, delimiter=self.__delimiter)
            n_sample = min (self.batch_size, self.train_number - i)
            train_data_this = train_data[i : i + n_sample]
            data = np.zeros ([self.gene_number, n_sample])

            for item in csv_reader:
                if csv_reader.line_num < 2:
                    continue
                if csv_reader.line_num > self.gene_number:
                    break
                nums = [float(n) for n in item[1: ]]
                nums_norm = (nums - np.min(nums)) / (np.max(nums) - np.min(nums))
                data[csv_reader.line_num - 2:, :] = nums_norm[train_data_this]
                if csv_reader.line_num % 1000 == 0:
                    print ('readline ', csv_reader.line_num)
            for j in range (n_sample):
                writelist = [self.sample_train[i + j], train_label[i + j]] + data.transpose()[j, :].tolist()
                write_csv.writerow (writelist)
            print ('reform done train data for batch %d, totally %d batche' % (i, self.train_number))
        write_file.close()

        write_file = open (self.test_filename, 'w', newline='')
        write_csv = csv.writer (write_file, delimiter=self.__delimiter)
        write_csv.writerow (['samples', 'label'] + self.gene)
        #reform test data
        for i in range (0, self.test_number, self.batch_size):
            csv_file.seek(0)
            csv_reader = csv.reader (csv_file, delimiter=self.__delimiter)
            n_sample = min (self.batch_size, self.test_number - i)
            test_data_this = test_data[i : i + n_sample]
            data = np.zeros ([self.gene_number, n_sample])

            for item in csv_reader:
                if csv_reader.line_num < 2:
                    continue
                if csv_reader.line_num > self.gene_number:
                    break
                nums = [float(n) for n in item[1: ]]
                nums_norm = (nums - np.min(nums)) / (np.max(nums) - np.min(nums))
                data[csv_reader.line_num - 2:, :] = nums_norm[test_data_this]
                if csv_reader.line_num % 1000 == 0:
                    print ('\treadline ', csv_reader.line_num)
            for j in range (n_sample):
                writelist = [self.sample_test[i + j], test_label[i + j]] + data.transpose()[j, :].tolist()
                write_csv.writerow (writelist)
            print ('reform done test data for batch %d, totally %d batche' % (i, self.test_number))
        write_file.close()


    def GetBatch_train(self):
        self.batch_end_train = False
        n_sample = min (self.batch_size, self.train_number - self.batch_indicator_train)
        label = self.train_label[self.batch_indicator_train: self.batch_indicator_train + n_sample]
        data = np.zeros ([n_sample, self.gene_number])
        for i in range (n_sample):
            while self.__csv_reader_train.line_num < 1:
                self.__csv_reader_train.__next__()
            item = self.__csv_reader_train.__next__()[2: ]
            nums = [float(n) for n in item]
            data[i, :] = nums
        self.batch_indicator_train += self.batch_size
        label_vector = np.zeros([n_sample, self.type_number])
        #print (label_vector.shape)
        #print (label.shape)
        for i in range (n_sample):
            if (i >= label_vector.shape[0] or i >= label.shape[0] or label[i] >= label_vector.shape[1]):
                print (i)
                print (label.shape)
                print (n_sample)
                print (self.train_label.shape)
                print (self.batch_indicator_train)
                print (i, ' ', int(label[i]), ' ', label_vector.shape, ' ', label.shape, ' ' )
            label_vector[i, int(label[i])] = 1
        if self.batch_indicator_train >= self.train_number:
            self.batch_end_train = True
            self.batch_indicator_train = 0
            self.__csv_file_train.seek(0)
            self.__csv_reader_train = csv.reader (self.__csv_file_train, delimiter=self.__delimiter)
        return (data, label_vector)

    def GetBatch_test(self):
        self.batch_end_test = False
        n_sample = min (self.batch_size, self.test_number - self.batch_indicator_test)
        label = self.test_label[self.batch_indicator_test: self.batch_indicator_test + n_sample]
        data = np.zeros ([n_sample, self.gene_number])
        for i in range (n_sample):
            while self.__csv_reader_test.line_num < 1:
                self.__csv_reader_test.__next__()
            item = self.__csv_reader_test.__next__()[2: ]
            nums = [float(n) for n in item]
            data[i, :] = nums
        self.batch_indicator_test += self.batch_size
        if self.batch_indicator_test >= self.test_number:
            self.batch_end_test = True
            self.batch_indicator_test = 0
            self.__csv_file_test.seek(0)
            self.__csv_reader_test = csv.reader (self.__csv_file_test, delimiter=self.__delimiter)
        label_vector = np.zeros([n_sample, self.type_number])
        for i in range (n_sample):
            label_vector[i, int(label[i])] = 1
        return (data, label_vector)

    def GetTrain_SameLabel (self, label):
        csv_file = open (self.train_filename)
        csv_reader = csv.reader (csv_file, delimiter=self.__delimiter)
        data = np.zeros ([0, self.gene_number])
        for item in csv_reader:
            if csv_reader.line_num < 2:
                continue
            if int(item[1]) == label:
                #print (data.shape)
                #print ( np.array([float(n) for n in item[2:]]).shape)
                data = np.append (data, np.array([[float(n) for n in item[2:]]]), axis=0)
        return data

    def GetTest_SameLabel (self, label):
        csv_file = open (self.test_filename)
        csv_reader = csv.reader (csv_file, delimiter=self.__delimiter)
        data = np.zeros ([0, self.gene_number])
        for item in csv_reader:
            if csv_reader.line_num < 2:
                continue
            if int(item[1]) == label:
                data = np.append (data, np.array([[float(n) for n in item[2:]]]), axis=0)
        return data


