def mainPartition(model_type,position):
    begin_time = time.time()
    drone_resources = pd.read_excel("D:/PycharmProjects/myGreatpy/drone-resources.xlsx")
  #  data = pd.read_excel("C:/Users/LJX/Desktop/other_models/resnet110-data.xlsx")
    if model_type == "balexnet":
        data = balexnet
    elif model_type == "bresnet":
        data = bresnet
    power = 3.5
    """==================-=========Population Initialization======================="""
    M = 1  
    # position = 3
    '''
    传统卷积神经网络：squeezenet-303  mobilenet-82  alexnet-23 resnet-322  b-alexnet-41  b-resnet-338
    '''
    # L = 338 #
    if model_type == "balexnet":
        L = 40
    elif model_type == "bresnet":
        L = 338
    LIND = (position + 1) * (L + 1)  
    Encoding = 'BG'
    problem = MyProblem(M,LIND,position,balexnet,bresnet,drone_resources,model_type,8,L,data)
    NIND = 5 
    p = 1import numpy as np
from sys import path as paths
from os import path
import os,warnings
import pandas as pd
import copy
warnings.filterwarnings("ignore")


class MyProblem(ea.Problem):
    def __init__(self,M,m,position,ba,br,drs,type,B,L,data):
        name = "model_partition"
        maxormins = [1] #List of target minimization markers, 1: Minimize the target; -1: Maximize the target
        Dim = m #
        varTypes = [1] * Dim  
        lb = [0] * Dim  
        ub = [1] * Dim  
        lbin = [1] * Dim 
        ubin = [1] * Dim 
        ea.Problem.__init__(self,name,M,maxormins,Dim,varTypes,lb,ub,lbin,ubin)
        self.position = position  #position number
        self.ba = ba  #b-alexnet
        self.br = br  #b-resnet
        self.drs = drs  #drone
        self.dim = L   
        # self.baLatency = None
        # self.brLatency = None
        # self.baDataSize = None
        # self.brDataSize = None
        self.baLatency_paleo = None
        self.brLatency_paleo = None
        self.baDataSize_paleo = None
        self.brDataSize_paleo = None
        '''
        CNN:alexnet resnet mobilenet squeezenet
        '''
        self.alexnet_latency_paleo = None
        self.alexnet_datasize_paleo = None
        self.resnet_latency_paleo = None
        self.resnet_datasize_paleo = None
        self.mobilenet_latency_paleo = None
        self.mobilenet_datasize_paleo = None
        self.squeezenet_latency_paleo = None
        self.squeezenet_datasize_paleo = None
        self.data = data 

      #  self.B = (B / 8) * 1024 * 1024e-3  
        self.B = (B / 8) * 1e+6
        self.ap = [4,14,18,26]  #b-alexnet
        self.rp = [2,11,119,127] #b-resnet
        if type == "balexnet":
            self.type = 1
        elif type == "bresnet":
            self.type = 2
      #  self.type = type  
    def aimFunc2(self,population):  
        


      #  print("Field:",population.Field)
        total_partition_pos,total_device_pos = [],[]  
        positions = []  
        objV = []
        for i in range(len(population.Chrom)):
            partition_pos,device_pos = self.calCChromToMChrom(population.Chrom[i])
            temp = []
            for j in range(len(partition_pos) - 1):
                temp.append(partition_pos[j][1])
            positions.append(temp)
            total_partition_pos.append(partition_pos)
            total_device_pos.append(device_pos)
        for i in range(len(total_partition_pos)):
            total_latency = 0
            for j in range(len(total_partition_pos[i])):
                latency = self.calModelLatency2(total_partition_pos[i][j],total_device_pos[i][j])
                total_latency += latency
            if self.type == 1:
                for j in range(len(positions[i])):
                    if positions[i][j] <= self.ap[0]:
                        total_latency += self.baDataSize_paleo[positions[i][j]] / self.B * 1e+3
                    elif positions[i][j] < self.ap[1]:
                        total_latency += (self.baDataSize_paleo[positions[i][j]] + self.baDataSize_paleo[
                            self.ap[0]]) / self.B * 1e+3
                    elif positions[i][j] == self.ap[1]:
                        total_latency += self.baDataSize_paleo[self.ap[0]] / self.B * 1e+3
                    elif positions[i][j] <= self.ap[2]:
                        total_latency += self.baDataSize_paleo[positions[i][j]] / self.B * 1e+3
                    elif positions[i][j] < self.ap[3]:
                        total_latency += (self.baDataSize_paleo[positions[i][j]] + self.baDataSize_paleo[
                            self.ap[2]]) / self.B * 1e+3
                    elif positions[i][j] == self.ap[3]:
                        total_latency += self.baDataSize_paleo[self.ap[2]] / self.B * 1e+3
                    else:
                        total_latency += self.baDataSize_paleo[positions[i][j]] / self.B * 1e+3
            elif self.type == 2:
                for j in range(len(positions[i])):
                    if positions[i][j] <= self.rp[0]:
                        total_latency += self.brDataSize_paleo[positions[i][j]] / self.B * 1e+3
                    elif positions[i][j] < self.rp[1]:
                        total_latency += (self.brDataSize_paleo[positions[i][j]] + self.brDataSize_paleo[self.rp[0]]) / self.B * 1e+3
                    elif positions[i][j] == self.rp[1]:
                        total_latency += self.brDataSize_paleo[self.rp[0]] / self.B * 1e+3
                    elif positions[i][j] <= self.rp[2]:
                        total_latency += self.brDataSize_paleo[positions[i][j]] / self.B * 1e+3
                    elif positions[i][j] < self.rp[3]:
                        total_latency += (self.brDataSize_paleo[positions[i][j]] + self.brDataSize_paleo[self.rp[2]]) / self.B * 1e+3
                    elif positions[i][j] == self.rp[3]:
                        total_latency += self.brDataSize_paleo[self.rp[2]] / self.B * 1e+3
                    else:
                        total_latency += self.brDataSize_paleo[positions[i][j]] / self.B * 1e+3
            objV.append(total_latency)
        res = [[obj] for obj in objV]
        res = np.array(res)
        population.ObjV = res  
        population.FitnV = -res 

    '''
    CNN：alexnet resnet mobilenet squeezenet
    '''
    def aimFunc2Traditional(self, population): 
        total_partition_pos,total_device_pos = [],[] 
        positions = [] 
        objV = []
        for i in range(len(population.Chrom)):
            partition_pos,device_pos = self.calCChromToMChrom(population.Chrom[i])
            temp = []
            for j in range(len(partition_pos) - 1):
                temp.append(partition_pos[j][1])
            positions.append(temp)
            total_partition_pos.append(partition_pos)
            total_device_pos.append(device_pos)
        for i in range(len(total_partition_pos)):
            total_latency = 0
            for j in range(len(total_partition_pos[i])):
                latency = self.calModelLatency2(total_partition_pos[i][j],total_device_pos[i][j])
                total_latency += latency
            if self.type == "alexnet":
                for j in range(len(positions[i])):
                    total_latency += self.alexnet_datasize_paleo[positions[i][j]] / self.B * 1e+3
            elif self.type == "resnet":
                for j in range(len(positions[i])):
                    total_latency += self.resnet_datasize_paleo[positions[i][j]] / self.B * 1e+3
            elif self.type == "mobilenet":
                for j in range(len(positions[i])):
                    total_latency += self.mobilenet_datasize_paleo[positions[i][j]] / self.B * 1e+3
            elif self.type == "squeezenet":
                for j in range(len(positions[i])):
                    total_latency += self.squeezenet_datasize_paleo[positions[i][j]] / self.B * 1e+3
            objV.append(total_latency)
        res = [[obj] for obj in objV]
        res = np.array(res)
        population.ObjV = res 
        population.FitnV = -res 

    def calCChromToMChrom(self,c_chrom):
        c_chrom_matrix = c_chrom.reshape((self.position + 1,self.dim + 1))
        partition,device_position = [],[]
        for i in range(len(c_chrom_matrix)):
            arr = np.where(c_chrom_matrix[i] == 1)[0]
            begin,end = arr[0],arr[-1]
            partition.append([begin,end])
        arr = []
        for i in range(len(partition)):
            arr.append(partition[i][0])
        partition_res = self.positionToPartition(arr)
        position = sorted(partition,key=(lambda x:x[0]))
        return position,partition_res

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



    def constrCond(self,idx_positions,device_positions,p):  
        flag = True
        idx_positions.sort()
        during = [[0,idx_positions[0]]]
        device_battery = self.drs['battery']
        for i in range(len(idx_positions) - 1):
            d = [idx_positions[i] + 1,idx_positions[i + 1]]
            during.append(d)
        during.append([idx_positions[len(idx_positions)-1] + 1,self.dim])
        for i in range(len(device_positions)):
            latency = self.calModelLatency2(during[device_positions[i]],i)
            need = (p * latency) / (5.0 * 3600)
            if need > float(device_battery[i]):
                flag = False
                break
        return flag


    def calLayerPredModel2(self):
        device_flops = self.drs['FLOPS']
        device_bandwidth = self.drs['IO']
        if self.type == 1:
            layer_type = self.ba['layer_type']
            flops = self.ba['FLOPs']
            mem_read = self.ba['MemRead']
            mem_write = self.ba['MemWrite']
            baLatency_paleo = []
            comm_data = []
            for i in range(self.position + 1):
                baLatency = self.eachLayterLatency2(layer_type,flops,mem_read,mem_write,device_flops[i],device_bandwidth[i])
                baLatency_paleo.append(baLatency)
            self.baLatency_paleo = baLatency_paleo
            for i in range(len(mem_write)):
                comm_data.append(mem_write[i])
            self.baDataSize_paleo = comm_data

        elif self.type == 2:
            layer_type = self.br['layer_type']
            flops = self.br['FLOPs']
            mem_read = self.br['MemRead']
            mem_write = self.br['MemWrite']
            brLatency_paleo = []
            comm_data = []
            for i in range(self.position + 1):
                brLatency = self.eachLayterLatency2(layer_type,flops,mem_read,mem_write,device_flops[i],device_bandwidth[i])
                t1, t2 = 0, 0
                for k in range(128, 128 + 7):
                    t1 += brLatency[k]
                for k in range(238, 238 + 7):
                    t2 += brLatency[k]
                del brLatency[238:238 + 7]
                del brLatency[128:128 + 7]
                brLatency.insert(128, t1)
                brLatency.insert(232, t2)
                brLatency_paleo.append(brLatency)
            for i in range(len(mem_write)):
                comm_data.append(mem_write[i])
            d1 = comm_data[128]
            d2 = comm_data[238]
            del comm_data[238:238 + 7]
            del comm_data[128:128 + 7]
            comm_data.insert(128, d1)
            comm_data.insert(232, d2)
            self.brLatency_paleo = brLatency_paleo
            self.brDataSize_paleo = comm_data

    def calLayerPredTraditionModel(self):
        device_flops = self.drs['FLOPS']
        device_bandwidth = self.drs['IO']
        if self.type == "alexnet":
            layer_type = self.data['layer_type']
            flops = self.data['FLOPs']
            mem_read = self.data['MemRead']
            mem_write = self.data['MemWrite']
            alexnet_latency_paleo = []
            comm_data = []
            for i in range(self.position + 1):
                latency = self.eachLayterLatency2(layer_type, flops, mem_read, mem_write, device_flops[i],
                                             device_bandwidth[i])
                alexnet_latency_paleo.append(latency)
            self.alexnet_latency_paleo = alexnet_latency_paleo
            for i in range(len(mem_write)):
                comm_data.append(mem_write[i])
            self.alexnet_datasize_paleo = comm_data
        elif self.type == "resnet":
            layer_type = self.data['layer_type']
            flops = self.data['FLOPs']
            mem_read = self.data['MemRead']
            mem_write = self.data['MemWrite']
            resnet_latency_paleo = []
            comm_data = []
            for i in range(self.position + 1):
                latency = self.eachLayterLatency2(layer_type, flops, mem_read, mem_write, device_flops[i],
                                             device_bandwidth[i])
                t1, t2 = 0, 0
                for k in range(111, 118):
                    t1 += latency[k]
                for k in range(221, 228):
                    t2 += latency[k]
                del latency[221:228]
                del latency[111:118]
                latency.insert(111, t1)
                latency.insert(221, t2)
                resnet_latency_paleo.append(latency)
            for i in range(len(mem_write)):
                comm_data.append(mem_write[i])
            d1 = comm_data[111]
            d2 = comm_data[221]
            del comm_data[221:228]
            del comm_data[111:118]
            comm_data.insert(111, d1)
            comm_data.insert(215, d2)
            self.resnet_latency_paleo = resnet_latency_paleo
            self.resnet_datasize_paleo = comm_data
        elif self.type == "mobilenet":
            layer_type = self.data['layer_type']
            flops = self.data['FLOPs']
            mem_read = self.data['MemRead']
            mem_write = self.data['MemWrite']
            mobilenet_latency_paleo = []
            comm_data = []
            for i in range(self.position + 1):
                latency = self.eachLayterLatency2(layer_type, flops, mem_read, mem_write, device_flops[i],
                                             device_bandwidth[i])
                mobilenet_latency_paleo.append(latency)
            self.mobilenet_latency_paleo = mobilenet_latency_paleo
            for i in range(len(mem_write)):
                comm_data.append(mem_write[i])
            self.mobilenet_datasize_paleo = comm_data
        elif self.type == "squeezenet":
            layer_type = self.data['layer_type']
            flops = self.data['FLOPs']
            mem_read = self.data['MemRead']
            mem_write = self.data['MemWrite']
            squeezenet_latency_paleo = []
            comm_data = []
            for i in range(self.position + 1):
                latency = self.eachLayterLatency2(layer_type, flops, mem_read, mem_write, device_flops[i],
                                             device_bandwidth[i])
                t = [0 for j in range(8)]
                total_k = [[5, 8], [11, 14], [17, 20], [24, 27], [30, 33], [36, 39], [42, 45], [49, 52]]
                total_insert = [5, 8, 11, 15, 18, 21, 24, 28]
                for k in range(len(total_k)):
                    for kk in range(total_k[k][0], total_k[k][1] + 1):
                        t[k] += latency[kk]
                for k in range(len(total_k)):
                    kk = len(total_k) - k - 1
                    del latency[total_k[kk][0]:total_k[kk][1] + 1]
                for k in range(len(total_k)):
                    latency.insert(total_insert[k], t[k])
                squeezenet_latency_paleo.append(latency)
            for i in range(len(mem_write)):
                comm_data.append(mem_write[i])
            d = [0 for i in range(len(total_k))]
            for k in range(len(total_k)):
                d[k] = comm_data[total_k[k][0]]
            for k in range(len(total_k)):
                kk = len(total_k) - k - 1
                del comm_data[total_k[kk][0]:total_k[kk][1] + 1]
            for k in range(len(total_k)):
                comm_data.insert(total_insert[k], d[k])
            self.squeezenet_latency_paleo = squeezenet_latency_paleo
            self.squeezenet_datasize_paleo = comm_data


    def eachLayterLatency2(self,layer_type,flops,mem_read,mem_write,device_flops,device_bandwidth):
        latency = []
        for i in range(len(layer_type)):
            input_comm_time = mem_read[i] / (device_bandwidth * 128 * 1024e-3)
            comp_time = flops[i] / (device_flops * 1e+9) * 1e+3
            output_comm_time = mem_write[i] / (device_bandwidth * 128 * 1024e-3)
            total_time = input_comm_time + comp_time + output_comm_time
            latency.append(total_time)
        return latency

    def calModelLatency2(self,positions,device_i):  #计算分割策略下子模型的推理延迟
        latency = 0
        if self.type == 1:
            for i in range(pre,next + 1):
                latency += self.baLatency_paleo[device_i][i]
        elif self.type == 2:
            for i in range(pre,next + 1):
                latency += self.brLatency_paleo[device_i][i]
        elif self.type == "alexnet":
            for i in range(pre,next + 1):
                latency += self.alexnet_latency_paleo[device_i][i]
        elif self.type == "resnet":
            for i in range(pre,next + 1):
                latency += self.resnet_latency_paleo[device_i][i]
        elif self.type == "mobilenet":
            for i in range(pre,next + 1):
                latency += self.mobilenet_latency_paleo[device_i][i]
        elif self.type == "squeezenet":
            for i in range(pre,next + 1):
                latency += self.squeezenet_latency_paleo[device_i][i]
        return latency

    def testSingleDeviceLatency(self):
        device_flops = self.drs['FLOPS']
        device_bandwidth = self.drs['IO']
        if self.type == 1:
            layer_type = self.ba['layer_type']
            flops = self.ba['FLOPs']
            mem_read = self.ba['MemRead']
            mem_write = self.ba['MemWrite']
            for i in range(len(device_flops)):
                latency = self.eachLayterLatency2(layer_type,flops,mem_read,mem_write, device_flops[i], device_bandwidth[i])
                total = 0
                for l in latency:
                    total += l
                print("latency:",total)
        elif self.type == 2:
            layer_type = self.br['layer_type']
            flops = self.br['FLOPs']
            mem_read = self.br['MemRead']
            mem_write = self.br['MemWrite']
            for i in range(len(device_flops)):
                latency = self.eachLayterLatency2(layer_type, flops, mem_read, mem_write, device_flops[i],
                                                  device_bandwidth[i])
                total = 0
                for l in latency:
                    total += l
                print("latency:", total)
