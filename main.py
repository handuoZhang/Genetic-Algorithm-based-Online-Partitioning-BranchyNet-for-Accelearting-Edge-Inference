import sys
sys.path.append("D:/PycharmProjects/myGreatpy")
import random
import time
import numpy as np
import geatpy as ea
import os
import pandas as pd
from itertools import permutations
from myGA.MutOper import myMutOper
from numpy import array

from myGA.MyProblem import MyProblem
from myGA.Xovmp import myXovmp
from myGA.soea_SEGA_myAlgorithm import soea_SEGA_myAlgorithm

'''
Obtain the data needed to calculate the objective function values and constraints
'''
b_time = time.time()
balexnet = pd.read_excel("C:/Users/Desktop/b-alexnet-data.xlsx")
bresnet = pd.read_excel("C:/Users/Desktop/b-resnet110-data.xlsx")
drone_resources = pd.read_excel("C:/Users/Desktop/drone-resources.xlsx")
data = pd.read_excel("C:/Users/Desktop/other_models/alexnet-data.xlsx")
e_time = time.time()
print("b_e_time:",(e_time - b_time)* 1e+3)
'''
Calculate the factorial of a number: calculate how many model deployments exist based on the number of devices that can deploy the model
'''
def factorial(num):
    factorial = 1
    if num < 0:
        return -1
    elif num == 0:
        return 1
    else:
        for i in range(1,num + 1):
            factorial = factorial * i
        return factorial
'''
After the model partitioning method is determined, all the model deployment methods under this partitioning strategy are calculated
'''
def calModelPermutation(num):
    res = list(permutations([i for i in range(0,num)],num))
    return res
'''
Generate the daughter chromosome corresponding to this parent chromosome based on the segmentation point position and c_chrom_idx
'''
def buildCChromMatrix(position_idx,c_chrom_idx,L):
    '''

    :param position_idx: position of partition，如[11,35]
    :param c_chrom_idx: Model Deployment Strategy：[(2,0,1),(0,2,1)....]
    :param L: Number of model layers-1
    :return: 1*(position + 1)* (L + 1) list 
    '''
    position_idx.sort()
    during = [[0,position_idx[0]]]
    for i in range(len(position_idx) - 1):
        d = [position_idx[i] + 1,position_idx[i + 1]]
        during.append(d)
    during.append([position_idx[len(position_idx)-1] + 1,L])
    res = []
    for i in range(len(c_chrom_idx)):
        arr = np.zeros((len(position_idx) + 1,L + 1),dtype=np.int8)
        for j in range(len(c_chrom_idx[i])):
            begin = during[c_chrom_idx[i][j]][0]
            end = during[c_chrom_idx[i][j]][1] + 1
            idx = [k for k in range(begin,end)]
            arr[j][idx] = 1
        arr = arr.flatten()
        res.append(arr.tolist())
    return res



    total_children_chrom = factorial(position + 1) * NIND
    c_chrom_size = total_children_chrom * p
    c_chrom_size = int(c_chrom_size)
    each_chrom_size = c_chrom_size / NIND  
    mod_chrom_size = c_chrom_size % NIND
    #print(random.sample(calModelPermutation(position + 1),int(each_chrom_size)))
    Field = np.zeros((NIND,L),dtype=np.int8)
    positions = []
    problem.calLayerPredModel2()
    c_Field = []def mainPartition(model_type,position):
    begin_time = time.time()
    drone_resources = pd.read_excel("D:/PycharmProjects/myGreatpy/drone-resources.xlsx")
  #  data = pd.read_excel("C:/Users/LJX/Desktop/other_models/resnet110-data.xlsx")
    if model_type == "balexnet":
        data = balexnet
    elif model_type == "bresnet":
        data = bresnet
    power = 3.5
    """==================-=========Population initialization========================"""
    M = 1  
    # position = 3
    '''
    CNN：squeezenet-303  mobilenet-82  alexnet-23 resnet-322  b-alexnet-41  b-resnet-338
    '''
    # L = 338 
    if model_type == "balexnet":
        L = 40
    elif model_type == "bresnet":
        L = 338
    LIND = (position + 1) * (L + 1) 
    Encoding = 'BG'
    problem = MyProblem(M,LIND,position,balexnet,bresnet,drone_resources,model_type,8,L,data)
    NIND = 5 
    p = 1
    permutation_res = calModelPermutation(position + 1)
    for i in range(len(Field)):
       # flag = True
        arr = Field[i]
        c_chroms_idx = []
        idx = random.sample(range(0, len(arr)), position)  
      #  idx = temp_partitions[i]
        print(idx)
        if mod_chrom_size != 0:
            device_position = random.sample(permutation_res, int(each_chrom_size + mod_chrom_size))
            mod_chrom_size = 0
        else:
            device_position = random.sample(permutation_res, int(each_chrom_size))  
        
       
        arr[idx] = 1
        Field[i] = arr   
        c_chroms = buildCChromMatrix(idx,device_position,L)
        c_Field.extend(c_chroms)
    c_Field = np.array(c_Field)
  #  print("c_Field:",c_Field)  
    population = ea.Population(Encoding,c_Field,c_chrom_size)
    population.Chrom = population.Field
    problem.aimFunc2(population)
    """===========================Algorithm parameter setting========================"""
    myAlgorithm = soea_SEGA_myAlgorithm(problem,population,position,L)
    myAlgorithm.MAXGEN = 200  
    myAlgorithm.timeSlot = time.time()  
    myAlgorithm.currentGen = 0 
    myAlgorithm.mutOper.Pm = 0.3 
    myAlgorithm.recOper.XOVR = 0.5  
    myAlgorithm.passTime = 0
    myAlgorithm.MAXTIME = None
    myAlgorithm.trappedCount = 0
    myAlgorithm.maxTrappedCount = 200
    myAlgorithm.trace = {"f_best":[],"f_avg":[]}
    myAlgorithm.trappedValue = 0 
    myAlgorithm.logTras = 1  
    myAlgorithm.log = {'gen':[],'eval':[]} if myAlgorithm.logTras != 0 else None
    myAlgorithm.verbose = True  
    myAlgorithm.drawing = 1 
    """========================Population Evolution==========================="""
    middle_time = time.time()
    begin_time = time.time()
    [BestIndi,population] = myAlgorithm.run(prophetPop=None,LIND=LIND,problem=problem,args=[L,power])
    """===========================Output Results========================"""
    BestIndi.save()
    if BestIndi.sizes != 0:
        print(BestIndi.Chrom)
        print(BestIndi.ObjV)
        partition_pos,device_pos = problem.calCChromToMChrom(BestIndi.Chrom[0])
        print(partition_pos)
        print(device_pos)
        return BestIndi.ObjV,partition_pos,device_pos
        # total_pw = power * BestIndi.ObjV[0][0]
        # print("total_pw:",total_pw)
    else:
        print(No viable solution found)
        return "No viable solution found"
    end_time = time.time()
    print("during_time:",(end_time - begin_time)* 1e+3)






