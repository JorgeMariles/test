import numpy as np
import random
from random import randint
import time
from pathlib import Path
import csv 
import glob
import os
#comentario para mis lectores de gitub
def cal_pop_fitness(items, population,d1,d2,number_bins,number_items,bin_size):
    # Calculating the fitness value of each solution in the current population.
    # using the falkenauer function and using k=2
    fitness=np.zeros((d1,d2)) 
    for k in range (d1):
        for z in range (d2):  
            fitness[k,z]=cal_pop_fitness_unit(items, population[k,z,:],number_bins,number_items,bin_size)
    return fitness

def cal_pop_fitness_unit(items, population,number_bins,number_items,bin_size):
    bins_used=np.unique(population)
    fi=np.zeros((1,np.shape(bins_used)[0]))
    for o in range(np.shape(bins_used)[0]):   
        index3=np.where(population==bins_used[o])
        count=np.sum(items[0,index3[0]])
        if count>bin_size:
            fi[0,o]=0
        else:
            fi[0,o]=(count/bin_size)**2
    fitness=np.sum(fi)/np.shape(bins_used)[0]
    return fitness

def crossover(fitness,population,equation_inputs,d1,d2,number_bins,number_items,bin_size,mr):
    #this function choses the best fitness of the neighbors and then convines both parents, if the solution has better fitness 
    #and it is viable it will replace the individual being tested
    new_population=np.zeros(np.shape(population))
    assert new_population.shape == population.shape
    for k in range(np.shape(population)[0]): 
        for z in range(np.shape(population)[1]):
            parent_1=population[k,z,:].copy()
            #l5 neighbors
            l=np.array([fitness[k,(z-1)%d2],fitness[(k-1)%d1,z],fitness[k,(z+1)%d2],fitness[(k+1)%d1,z]]) 
            selector=np.random.choice(np.arange(4), size=2, replace=False)
            s_v=np.array([selector[0],selector[1]])
            fit=[l[selector[0]],l[selector[1]]]
            index=np.where(fit==np.max(fit))
            index=list(zip(*index))[0]
            winner=s_v[index]
            #print(parent_1,"parent1")
            if winner==0:
                parent_2=population[k,(z-1)%d2,:].copy()
            elif winner==1:    
                parent_2=population[(k-1)%d1,z,:].copy()
            elif winner==2:
                parent_2=population[k,(z+1)%d2,:].copy()
            elif winner==3:
		parent_2=population[(k+1)%d1,z,:].copy()                
            prepop=offspring_op(parent_2,parent_1,population)    
            #print(parent_2,"parent_2")
            #print(prepop,"preop")z
            #print(k,z,"k,z")
            prepop=mutation(prepop,mr,number_bins,items,number_items,bin_size)
            if cal_pop_fitness_unit(items,prepop,number_bins,number_items,bin_size)>fitness[k,z]: 
                new_population[k,z,:]=prepop
            else:
                new_population[k,z,:]=parent_1 
            #print(new_population[k,z,:],"fianl")  
            #print("----------------------")              
    return new_population

#convines the parents 
def offspring_op(parent_2,parent_1,population):
    rnd=np.random.uniform(low=0, high=1)
    max_crosover =np.uint8(np.shape(population)[2])
    crossover_point=np.random.randint(low=1,high=max_crosover-1)
    if rnd >.5:
        offspring = parent_1[0:crossover_point]
        offspring = np.concatenate((offspring,parent_2[crossover_point:]),0)
    else:
        offspring = parent_2[0:crossover_point]
        offspring = np.concatenate((offspring,parent_1[crossover_point:]),0)    
    return offspring


#mutates a random gene with a random bin if the random number generator is below or equal to the mutation rate, then checks 
#if that solution is posible, if it is the gene is changed otherwise it reamins the same
def mutation(population,mr,number_bins,items,number_items,bin_size): 
    if np.random.uniform(low=0, high=1)<=mr:
                #print(k,z,"mutation")
                # The random bin to be changed in the genome.
        random_bin = np.random.randint(low=1, high=number_bins)
        cromo = randint(0, np.shape(population)[0]-1)               
        gene=population.copy()               
        gene[cromo]=random_bin               
        population=gene                                                     
    return population


#bin packaging creation
def population_creation(pop_size,number_bins,number_items,bin_size):
    population=np.zeros(pop_size)  
    for k in range(np.shape(population)[0]):
        for z in range(np.shape(population)[1]):
            population[k,z,:]=np.random.randint(low=1, high=number_bins+1,size=population.shape[2])      
    return population           


fit_prom=[]
time_prom=[]
generations_prom=[]
ide=[]
bins=[]
count=0
direc=r"C:\Users\monit\Documents\Tec\CGA\BPP instances\Augmented_Non_IRUP_and_Augmented_IRUP_Instances\Difficult_Instances\ANI"
for l in os.listdir(direc):
    #bin Packaging 1d
    print(l)
    data_folder = Path(direc)
    file_to_open =str(data_folder /l)
    File = open(file_to_open,'r').readlines()
    number_bins=int(File[0])
    bin_size=int(File[1])
    number_items=len(File)-2
    items=np.zeros((1,number_items))
    for x in range(2,len(File)):
        items[0,x-2]=int(File[x])
      
    for z in range (10):
        start_time = time.time()
        #dimension of the population matrix, mutation rate and number of generations
        d1=16
        d2=16
        mr=.5
        stop_treshold=20
        # Defining the population size.
        pop_size = (d1,d2,(number_items)) 
        #Creating the initial population.
        population=population_creation(pop_size,number_bins,number_items,bin_size)
        elapsed_time = time.time() - start_time
        #print(population)
        x1=[]
        y1=[]
        generation=0
        treshold=0
        best_pre=0
        while treshold<=stop_treshold:   
            fitness = cal_pop_fitness(items,population,d1,d2,number_bins,number_items,bin_size)
            #a.append(fitness)
            best=np.max(fitness)
            #print(best)
            index=np.where(fitness==best)
            x1.append(generation)
            y1.append(np.max(fitness))
            index=list(zip(*index))[0]
            index=list(index)
            bins_used=np.shape(np.unique(population[index[0],index[1],:]))
            
            # Generating next generation using crossover.
            new_population = crossover(fitness,population,items,d1,d2,number_bins,number_items,bin_size,mr)
            #Adding some variations to the offsrping using mutation.    
            population=new_population
            if (best-best_pre)<.001:
                treshold=treshold+1
            else:
                treshold=0
            generation=generation+1
            best_pre=best
        elapsed_time2= time.time() - start_time
        total_time=elapsed_time+elapsed_time2
        total_time=total_time*1000
        fitness_result = cal_pop_fitness(items,population,d1,d2,number_bins,number_items,bin_size)
        x1.append(generation)
        y1.append(np.max(fitness))
        bins_used=np.shape(np.unique(population[index[0],index[1],:]))
	bins.append(bins_used)
        best=np.max(fitness)
        print(total_time)
        print(best)
        index1=np.where(fitness==best)
        fit_prom.append(best)
        time_prom.append(total_time)
        generations_prom.append(generation)
        ide.append(l)
    print("finish",l)    
    count=count+1
    print(count)  

name="Difficult_Instances/ANI"
with open(name, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow (["class","id", "fitness", "time ms","generations","min bins"])
    for x in range (len(ide)):
         writer.writerow([name,ide[x], fit_prom[x], time_prom[x],generations_prom[x],bins[x]])



