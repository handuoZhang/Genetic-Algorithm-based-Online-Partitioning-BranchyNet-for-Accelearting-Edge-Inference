import copy
import random

import geatpy as ea
import numpy as np
from geatpy.operators.recombination.Recombination import *

class myMutOper(Recombination):
    def __init__(self, Pm=None, Parallel=False,position = None,L = None):
        self.Pm = Pm  # The probability of mutation of the smallest segment on the chromosome on which the mutation operator acts
        self.Parallel = Parallel  
        self.position = position  
        self.dim = L 

    def do(self,Encoding,OldChrom,FieldDR,*args):
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
            total_MChrom.append(arr)
        #print("total_MChrom:",total_MChrom)
        diff_total_MChrom = list(set([tuple(t) for t in total_MChrom]))
        diff_total_MChrom = [list(v) for v in diff_total_MChrom]
        for i in range(len(diff_total_MChrom)):
            r = random.random()
            if r < self.Pm:
                one_idx = [idx for idx in range(len(diff_total_MChrom[i])) if diff_total_MChrom[i][idx] == 1]
                zero_idx = [idx for idx in range(len(diff_total_MChrom[i])) if diff_total_MChrom[i][idx] == 0]
                r1 = random.randint(0,len(one_idx) - 1)
                r2 = random.randint(0,len(zero_idx) - 1)

                new_MChrom = np.zeros((1,len(diff_total_MChrom[i])),dtype=np.int8)
                new_MChrom[0][one_idx] = 1
                new_MChrom = new_MChrom[0].tolist()
                new_MChrom[one_idx[r1]] = 0
                new_MChrom[zero_idx[r2]] = 1

                one_idx = [idx for idx in range(len(new_MChrom)) if new_MChrom[idx] == 1]
                one_idx.sort()
                during = [[0, one_idx[0]]]
                for j in range(len(one_idx) - 1):
                    d = [one_idx[j] + 1, one_idx[j + 1]]
                    during.append(d)
                during.append([one_idx[len(one_idx) - 1] + 1, self.dim])

                index = [k for k in range(len(total_MChrom)) if all(total_MChrom[k] == diff_total_MChrom[i])]  #[0~n]
                for obj in index:
              
                    arr = np.zeros((self.position + 1, self.dim + 1), dtype=np.int8)
                    for j in range(len(total_device_pos[obj])):
                        begin = during[total_device_pos[obj][j]][0]
                        end = during[total_device_pos[obj][j]][1] + 1
                        idx = [k for k in range(begin, end)]
                        arr[j][idx] = 1
                    arr = arr.flatten()
                    newChrom[obj] = arr.tolist()
            else:
                index = [k for k in range(len(total_MChrom)) if all(total_MChrom[k] == diff_total_MChrom[i])]
                for obj in index:
                    newChrom[obj] = OldChrom[obj]
        newChrom = np.array(newChrom)
        for i in range(n):
            partition_pos, device_pos = self.calCChromToMChrom(newChrom[i])
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
