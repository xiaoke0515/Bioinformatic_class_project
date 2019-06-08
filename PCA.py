import numpy as np
import csv
from sklearn.decomposition import IncrementalPCA 
import pathlib
from sparse import DataReader, PCAResultReader
import pickle

def PCA(reduced_component=50, batch_size=100):
    # initialize ipca reader classreader
    datafile = "Gene_Chip_Data/Gene_Chip_Data/microarray.original.txt"
    ipca = IncrementalPCA(n_components=reduced_component, batch_size=batch_size)
    reader = DataReader (filename=datafile, batch_size=100, delimiter='\t')

    # fit ipca
    print ("fitting~!")
    while not reader.batch_end:
        data = reader.GetBatch()
        ipca.partial_fit(data.transpose())
        print ("finish fitting batch %d, totally %d !" % (reader.batch_indicator, reader.sample_number))
        print (data)
        print (data.shape)
    reader.Reset()

    # get engenvalue and eigenvector
    eigenvalue = ipca.explained_variance_
    eigenvector = ipca.components_
    component_variance = ipca.components_
    singular = ipca.singular_values_
    mean_ = ipca.mean_
    var_ = ipca.var_
    noise_ = ipca.noise_variance_
    print ("eigenvalue shape", eigenvector.shape)
    print (eigenvalue)
    print (eigenvector)
    picklefile = open('Gene_Chip_Data/Gene_Chip_Data/pca_artibutes.txt', 'wb')
    pickle.dump([eigenvalue, eigenvector, component_variance, singular, mean_, var_, noise_], picklefile)
    picklefile.close()


    # transform 
    print ("transforming!")
    transform_results = np.zeros([reduced_component, reader.sample_number])
    precisions = np.zeros ([reader.sample_number])
    while not reader.batch_end:
        n_sample = min (batch_size, reader.sample_number - reader.batch_indicator)
        indicator = reader.batch_indicator
        #print ('btchsis', batch_size)
        #print ('number', reader.sample_number)
        #print ('indicator', reader.batch_indicator)
        print ('nsample', n_sample)
        data = reader.GetBatch()[:, :n_sample]
        #print ('datashape', data.shape)
        res = ipca.transform(data.transpose())
        #print ('resshape', res.shape)
        #print (transform_results[:, indicator:indicator + n_sample].shape)
        transform_results[:, indicator: indicator + n_sample] = res.transpose()
        # get precision
        #reformed = np.matmul(eigenvector.transpose(), res.transpose())
        reformed = ipca.inverse_transform (res).transpose()
        precision = np.sum (np.square(reformed - data), 0)
        precisions[indicator: indicator + n_sample] = precision
        print ("finish transforming batch %d, totally %d !" % (reader.batch_indicator, reader.sample_number))
        #print (" the batch" , reader.batch_indicator)
        #print (res)
        #print (data)
        #print (reformed)
        #print (precision)

    print (transform_results.shape)

    # print precision 
    print ("the total loss is %f" % np.sum(precisions))


    csvFile = open("Gene_Chip_Data/Gene_Chip_Data/pca_result.txt", 'w')
    writer = csv.writer (csvFile)
    writer.writerow (reader.sample)
    writer.writerows (transform_results)
    csvFile.close()

def GetPCAResult(reduced_component=50, batch_size=100):
    if not pathlib.Path("Gene_Chip_Data/Gene_Chip_Data/pca_result.txt").is_file():
        PCA(reduced_component=reduced_component, batch_size=batch_size)
    pcareader = PCAResultReader ("Gene_Chip_Data/Gene_Chip_Data/pca_result.txt", ',')
    if pcareader.factor_number != reduced_component:
        del pcareader
        PCA(reduced_component=reduced_component, batch_size=batch_size)
        pcareader = PCAResultReader ("Gene_Chip_Data/Gene_Chip_Data/pca_result.txt", ',')
    return pcareader

if __name__ == '__main__':
    GetPCAResult()
