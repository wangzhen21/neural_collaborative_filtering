'''
Created on Aug 8, 2016
Processing datasets. 

@author: Xiangnan He (xiangnanhe@gmail.com)
'''
import scipy.sparse as sp
import numpy as np
from tqdm import tqdm

class Dataset(object):
    '''
    classdocs
    '''

    def __init__(self, path):
        '''
        Constructor
        '''
        self.trainMatrix = self.load_rating_file_as_matrix(path + ".train.rating")
        self.testRatings = self.load_rating_file_as_list(path + ".test.rating")
        self.testNegatives = self.load_negative_file(path + ".test.negative")
        assert len(self.testRatings) == len(self.testNegatives)
        
        self.num_users, self.num_items = self.trainMatrix.shape
        
    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        return ratingList
    
    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1: ]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList
    
    def load_rating_file_as_matrix(self, filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        # Construct matrix
        mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if (rating > 0):
                    mat[user, item] = 1.0
                line = f.readline()    
        return mat
class Datasetstance(object):
    '''
    classdocs
    '''

    def __init__(self, path):
        '''
        Constructor
        '''
        self.trainMatrix,self.trainstanceMatrix,self.stance_num = self.load_rating_file_as_matrix(path + ".train.rating.stance")
        self.stance_num += 1
        self.testRatings = self.load_rating_file_as_list(path + ".test.rating.stance")
        self.testNegatives,self.testNegativesDre = self.load_negative_file(path + ".test.negtive.stance.newindex",path + ".test.negtive.flag.stance.newindex")
        assert len(self.testRatings) == len(self.testNegatives)

        self.num_users, self.num_items = self.trainMatrix.shape

    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item,stance = int(arr[0]), int(arr[1]),int(arr[5])
                if stance > self.stance_num:
                    stance = self.stance_num
                ratingList.append([user, item,stance])
                line = f.readline()
        return ratingList

    def load_negative_file(self, filename,filename2):
        negativeList = []
        negativedireList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1:]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        with open(filename2, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1:]:
                    negatives.append(int(x))
                negativedireList.append(negatives)
                line = f.readline()
        return negativeList,negativedireList

    def load_rating_file_as_matrix(self, filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        num_users, num_items,num_stance = 0, 0,0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                u, i,s = int(arr[0]), int(arr[1]),int(arr[5])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                num_stance = max(num_stance,s)
                line = f.readline()
        # Construct matrix
        mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
        matstance = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
        iline = 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                i+=1
                if i%10000 == 0:
                    print i
                arr = line.split("\t")
                user, item, rating,stance = int(arr[0]), int(arr[1]), float(arr[2]),int(arr[5])
                if (rating > 0):
                    mat[user, item] = 1.0
                    matstance[user, item] = stance
                line = f.readline()
        return mat,matstance,num_stance