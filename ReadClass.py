import numpy as np
from sparse import *

def GetClass():
    class_file = 'Gene_Chip_Data/Gene_Chip_Data/E-TABM-185.sdrf.txt'
    reader = ClassReader (class_file, delimter='\t')
    #print (reader.type_number)
    reader.PrintTypeNumber()
    return reader

if __name__ == '__main__':
    GetClass()
