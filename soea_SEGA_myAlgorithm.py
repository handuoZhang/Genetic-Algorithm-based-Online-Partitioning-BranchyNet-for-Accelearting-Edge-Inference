# -*- coding: utf-8 -*-
import os.path
import random
import time
import pandas as pd
import geatpy as ea  
from sys import path as paths
from os import path
import geatpydev.geatpy as myea
import numpy as np

from myGA.MutOper import myMutOper
from myGA.Xovmp import myXovmp


class soea_SEGA_myAlgorithm(myea.SoeaAlgorithm):

    def __init__(self,problem,population,position,L):
        ea.SoeaAlgorithm.__init__(self,problem,population)   
        if population.ChromNum != 1:
            raise RuntimeError('The incoming population object must be a single chromosome population type.')
        self.name = 'SEGA'
        self.selFunc = 'tour'  
        self.population = population
        self.drawing = 1  
        self.recOper = myXovmp(XOVR=0.7,Half_N=False,GeneID=None,Parallel=False,position=position,L=L)   
        self.mutOper = myMutOper(Pm=0.5,Parallel=False,position=position,L=L)   
    def stat(self,pop,problem,args):
        feasible = []
        Chrom = pop.Chrom
        for i in range(len(Chrom)):
            partition_pos,device_pos = problem.calCChromToMChrom(Chrom[i])
            temp = []
            for j in range(len(partition_pos) - 1):
                temp.append(partition_pos[j][1])
            isSatisfy = problem.constrCond(temp,device_pos,args[1])
            if isSatisfy:
                feasible.append(i)
        if len(feasible) > 0:
            feasiblePop = pop[feasible]
            bestIndi = feasiblePop[np.argmax(feasiblePop.FitnV)]
            if self.BestIndi is None:
                self.BestIndi = bestIndi   
            else:
                delta = (self.BestIndi.ObjV - bestIndi.ObjV) * self.problem.maxormins if self.problem.maxormins is not None else self.BestIndi.ObjV - bestIndi.ObjV
                self.trappedCount += 1 if np.abs(delta) < self.trappedValue else 0
                if delta > 0:
                    self.BestIndi = bestIndi
            self.trace['f_best'].append(bestIndi.ObjV[0][0])
            self.trace['f_avg'].append(np.mean(feasiblePop.ObjV))
            if self.logTras != 0 and self.currentGen % self.logTras == 0:
                self.logging(feasiblePop)  
                if self.verbose:
                    self.display() 
            return bestIndi.ObjV[0][0]
        return None

    def terminated(self, pop, problem, args,generation_end_list):
        generation_bestIndi = self.stat(pop, problem, args)  
        self.passTime += time.time() - self.timeSlot  
        self.timeSlot = time.time()  
        if len(generation_end_list) == 50:
            return True
        else:
            if len(generation_end_list) == 0:
                generation_end_list.append(generation_bestIndi)
            elif generation_end_list[-1] != generation_bestIndi:
                generation_end_list.clear()
                generation_end_list.append(generation_bestIndi)
            else:
                generation_end_list.append(generation_bestIndi)
            self.currentGen += 1
            return False



    def finishing(self,pop,problem,args):

        feasible = []
        Chrom = pop.Chrom
        for i in range(len(Chrom)):
            partition_pos,device_pos = problem.calCChromToMChrom(Chrom[i])
            temp = []
            for j in range(len(partition_pos) - 1):
                temp.append(partition_pos[j][1])
            isSatisfy = problem.constrCond(temp,device_pos,args[1])
            if isSatisfy:
                feasible.append(i)
        if len(feasible) > 0:
            feasiblePop = pop[feasible]
            if self.logTras != 0 and (len(self.log['gen']) == 0 or self.log['gen'][-1] != self.currentGen): 
                self.logging(feasiblePop)
                if self.verbose:
                    self.display()
        self.passTime += time.time() - self.timeSlot  
       # self.draw(pop, EndFlag=True)  
        return [self.BestIndi, pop]

    def call_aimFunc(self,pop,problem):
        #problem.aimFunc2(population=pop)
        problem.aimFunc2Traditional(population=pop)

    def run(self,prophetPop=None,LIND = None,problem = None,args = None):   
        population = self.population
        NIND = population.sizes 
        population.Lind = LIND
        if prophetPop is not None:
            population = (prophetPop + population)[:NIND]  
        max_history_idx = np.argmax(population.FitnV)
        max_history_fit = population[max_history_idx].FitnV[0][0]
        generation_end_list = []
        while self.terminated(population,problem,args,generation_end_list) == False:
            offspring = population[ea.selecting(self.selFunc,population.FitnV,NIND)]
            n = len(offspring.Chrom)
            total_MChrom = []
            for i in range(n):
                partition_pos,device_pos = problem.calCChromToMChrom(offspring.Chrom[i])
                temp = []
                for j in range(len(partition_pos) - 1):
                    temp.append(partition_pos[j][1])
                arr = np.zeros((1,args[0]),dtype=np.int8)
                arr = arr.flatten()
                arr[temp] = 1
                total_MChrom.append(arr.tolist())

            diff_total_MChrom = list(set([tuple(t) for t in total_MChrom]))
            diff_total_MChrom = [list(v) for v in diff_total_MChrom]
            m = len(diff_total_MChrom)
            next_population_chrom = []
            if m % 2 != 0:
                if m > 1:
                    odd_mchrom = random.sample(diff_total_MChrom,1)
                   # even_mchrom = list(set(diff_total_MChrom).difference(set(odd_mchrom)))
                    even_mchrom = []
                    for obj in diff_total_MChrom:
                        if obj != odd_mchrom[0]:
                            even_mchrom.append(obj)
                    odd_cchrom = [offspring.Chrom[i] for i in range(n) if total_MChrom[i] == odd_mchrom]
                    even_cchrom = [offspring.Chrom[i] for i in range(n) if total_MChrom[i] in even_mchrom]
                    c_chrom = self.recOper.do(even_cchrom)
                    next_population_chrom.extend(odd_cchrom)
                    next_population_chrom.extend(c_chrom)
                else:
                    next_population_chrom.extend(offspring.Chrom)
            else:
                chrom = self.recOper.do(offspring.Chrom)
                next_population_chrom.extend(chrom)
            next_population_chrom = np.array(next_population_chrom)
            offspring.Chrom = self.mutOper.do(Encoding=None,OldChrom=next_population_chrom,FieldDR=None)
            self.call_aimFunc(pop=offspring,problem=problem)
            max_idx = np.argmax(offspring.FitnV)
            min_idx = np.argmin(offspring.FitnV)
            # if offspring[max_idx].FitnV[0][0] >= max_history_fit:
            #     max_history_idx = max_idx
            #     max_history_fit = offspring[max_history_idx].FitnV[0][0]
            # else:
            #     offspring[min_idx] = population[max_history_idx]
            #     self.call_aimFunc(pop=offspring,problem=problem)
            #     max_history_idx = np.argmax(offspring.FitnV)
            # population = offspring
            # self.call_aimFunc(pop=population,problem=problem)
            population = population + offspring
            self.call_aimFunc(pop=population,problem=problem)
            population = population[ea.selecting('etour',population.FitnV,NIND)]
        return self.finishing(population,problem,args)

    def draw(self, pop, EndFlag=False):
        if not EndFlag:
            self.passTime += time.time() - self.timeSlot  
            if self.drawing == 2:
                metric = np.array(self.trace['f_best']).reshape(-1, 1)
                self.ax = ea.soeaplot(metric, Label='Objective Value', saveFlag=False, ax=self.ax, gen=self.currentGen,
                                      gridFlag=False)  
            elif self.drawing == 3:
                self.ax = ea.varplot(pop.Phen, Label='decision variables', saveFlag=False, ax=self.ax,
                                     gen=self.currentGen, gridFlag=False)
            self.timeSlot = time.time()  
        else:

            if self.drawing != 0:
                metric = np.array(self.trace['f_best']).reshape(-1,1)
                # if os.path.exists("C://Users//LJX//Desktop//data//alexnet//N5_p02.csv"):
                #     df1 = pd.read_csv("C://Users//LJX//Desktop//data//alexnet//N5_p02.csv")
                #     df2 = pd.DataFrame(metric)
                #     df = df1.join(df2)
                #     df.to_csv("C://Users//LJX//Desktop//data//alexnet//N5_p02.csv",index=False,mode='w',sep=',')
                # else:
                #     df = pd.DataFrame(metric)
                #     df.to_csv("C://Users//LJX//Desktop//data//alexnet//N5_p02.csv",index=False)
                ea.trcplot(metric, [['Value of population optimal individual objective function']], xlabels=[['Number of Generation']],ylabels=[['Value']], gridFlags=[[False]])


