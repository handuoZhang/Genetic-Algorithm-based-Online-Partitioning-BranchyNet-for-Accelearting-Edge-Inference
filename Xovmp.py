import copy
import random

import geatpy as ea
import numpy as np
from geatpy.operators.recombination.Recombination import *

class myXovmp(Recombination):

    def __init__(self, XOVR=None, Half_N=False, GeneID=None, Parallel=False,position = None,L = None):
        self.XOVR = XOVR  # Probability of occurrence of crossover
        self.Half_N = Half_N  
        self.GeneID = GeneID  # ID
        self.Parallel = Parallel 
        self.position = position  #position number
        self.dim = L 

    def do(self,OldChrom):
        n = len(OldChrom)
        newChrom = [[] for i in range(n)]
        total_MChrom = [] 
        total_partition_pos,total_device_pos = [],[]
        positions = [] 
        for i in range(n):
            partition_pos,device_pos = self.calCChromToMChrom(OldChrom[i])
            temp = []
            for j in range(len(partition_pos) - 1):
                temp.append(partition_pos[j][1])
            positions.append(temp)
            total_partition_pos.append(partition_pos)
            total_device_pos.append(device_pos)
            arr = np.zeros((1,self.dim),dtype=np.int8)
            arr = arr.flatten()
            arr[temp] = 1
            total_MChrom.append(arr.tolist())
        #print("total_MChrom:",total_MChrom)
        diff_total_MChrom = list(set([tuple(t) for t in total_MChrom]))
        diff_total_MChrom = [list(v) for v in diff_total_MChrom]
       # print("diff_total_MChrom:",diff_total_MChrom)
        m = len(diff_total_MChrom)
        before_xovmp,after_xovmp = [],[]
        group1 = diff_total_MChrom[0:int(m/2)]
        group2 = diff_total_MChrom[int(m/2):]
        combined = zip(group1,group2)
        for f1,f2 in combined:
            before_xovmp.append(f1)
            before_xovmp.append(f2)
            r = random.random()
            if r < self.XOVR: 
                f1_idx = [idx for idx in range(len(f1)) if f1[idx] == 1]
                f2_idx = [idx for idx in range(len(f2)) if f2[idx] == 1]
                f1_new = np.zeros((1,len(f1)),dtype=np.int8)
                f2_new = np.zeros((1,len(f2)),dtype=np.int8)
                f1_new[0][f1_idx] = 1
                f2_new[0][f2_idx] = 1
                f1_new = f1_new[0].tolist()
                f2_new = f2_new[0].tolist()
                union = list(set(f1_idx).intersection(set(f2_idx)))
                f1_idx = [idx for idx in f1_idx if idx not in union]
                f2_idx = [idx for idx in f2_idx if idx not in union]
                r1 = random.sample(range(0,len(f1_idx)),1)
                f1_new[f1_idx[r1[0]]] = 0
                f2_new[f1_idx[r1[0]]] = 1
                r2 = random.sample(range(0,len(f2_idx)),1)
                f2_new[f2_idx[r2[0]]] = 0
                f1_new[f2_idx[r2[0]]] = 1
                after_xovmp.append(f1_new)
                after_xovmp.append(f2_new)
            else:
                after_xovmp.append(f1)
                after_xovmp.append(f2)
        for i in range(n):
            index = before_xovmp.index(total_MChrom[i])
            mchrom = after_xovmp[index]
            
            if mchrom != before_xovmp[index]:
                one_idx = [idx for idx in range(len(mchrom)) if mchrom[idx] == 1]
                one_idx.sort()
                during = [[0,one_idx[0]]]
                for j in range(len(one_idx) - 1):
                    d = [one_idx[j] + 1,one_idx[j + 1]]
                    during.append(d)
                during.append([one_idx[len(one_idx) - 1] + 1,self.dim])
                arr = np.zeros((self.position + 1,self.dim + 1),dtype = np.int8)
                for j in range(len(total_device_pos[i])):
                    begin = during[total_device_pos[i][j]][0]
                    end = during[total_device_pos[i][j]][1] + 1
                    idx = [k for k in range(begin,end)]
                    arr[j][idx] = 1
                arr = arr.flatten()
                newChrom[i] = arr.tolist()
            else:
                newChrom[i] = OldChrom[i]
        newChrom = np.array(newChrom)
        return newChrom


    def calCChromToMChrom(self, c_chrom):
        c_chrom_matrix = c_chrom.reshape((self.position + 1, self.dim + 1))
        partition, device_position = [], []
        for i in range(len(c_chrom_matrix)):
            arr = np.where(c_chrom_matrix[i] == 1)[0]
            begin, end = arr[0], arr[-1]
            partition.append([begin, end])
        arr = []
        for i in range(len(partition)):
            arr.append(partition[i][0])
        partition_res = self.positionToPartition(arr)
        position = sorted(partition, key=(lambda x: x[0]))
        return position, partition_res

    def positionToPartition(self,partition):
        l2 = copy.deepcopy(partition)
        result = list()
        d = dict()
        partition.sort()
        for index,value in enumerate(partition):
            d[value] = index
        for i in l2:
            result.append(d[i])
        return result

