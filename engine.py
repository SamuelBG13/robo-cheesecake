# -*- coding: utf-8 -*-
"""
engine.py:
    This script generates the engine for our ProMPs distro.
    
"""
######################################################################################################

import numpy as np 
import numpy.random as npr
import pandas as pd
import matplotlib.pyplot as plt


#########

#%%
def GenerateToyData(): # Generates first a times vector
    ##
    def GenerateTimesVector(rate=100, maxtime=3): #Generates a vector of times. Rate is data/sec
        maxtime=maxtime+npr.uniform (-0.5,0.5) # Solutions vary uniformingly around the input value, 3 seconds in this case
        stampnum=np.round(maxtime*rate)  #stampnum is the number of timestamps at the demonstrations
        increase=maxtime/stampnum
        Timevec=np.arange(0,stop=stampnum/rate, step=increase)
        return stampnum, Timevec
    ##
    def GenerateAngles(Timevec): # Creates joint space timestamps based on a time vector 
        baselineth=np.array([0,0,0,-np.pi/2,0,np.pi/2, 0]) # A baseline for generating the thetas.
        q=np.array([])
        variation= npr.normal(0, 0.1 ,7) #The whole distribution shifts with gaussian noise
        for i, timestamp in enumerate(Timevec):
            anglesstamp=baselineth+variation+(((1+timestamp)**3)/80) #Gets angles numbers that increase with time
            if i==0:
                q=anglesstamp
            else:
                q=np.vstack((q, anglesstamp))
        return q
    ##
        
    N=3 #Number of demonstrations
    columns=['Times', 'q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6']
    df = pd.DataFrame(columns=columns) #Empty dataframe
    for n in range(N): #Creates N demonstrations, including the times vector and the angles  
        stampnum, Timevec=GenerateTimesVector()
        q=GenerateAngles(Timevec)
        df=df.append(pd.DataFrame(data={'Times': [Timevec], \
                                        'q0': [q[:,0]], \
                                        'q1': [q[:,1]], \
                                        'q2': [q[:,2]], \
                                        'q3': [q[:,3]], \
                                        'q4': [q[:,4]], \
                                        'q5': [q[:,5]], \
                                        'q6': [q[:,6]]}),ignore_index=True)
        

    return df
##

def GeneratePlot(df):
    Qlist=['q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6'] #list of q strings 
    plt.close("all") 
    plt.figure() # The generated data for each joint


    for index, row in df.iterrows():
        for q in range(7):
            plt.subplot(1,7,q+1)
            
            plt.plot(row["Times"], row[Qlist[q]])
            plt.title(Qlist[q])
            plt.ylim(-np.pi,np.pi)
            plt.suptitle('Toy demonstrated data')
            
            
##
df=GenerateToyData()
params = {'D' : 7, 'K' : 15, 'N' : df.shape[0]}
GeneratePlot(df)
    

   
#%%
     
#############################
    

# ProMP is the main class of the engine.  
# The constructor method requires two inputs:
# - A string Identifier which roughly describes the task in hand
# - A dataframe TrainingData which contains all the examples for the robot. 
# - A dictionary containing parameters: DoF number, number of basis functions, number of demonstrations  → (D,K,N)
# 
# The dataframe should include one row for each demonstration, which should have: 
# - A NumPy vector of sampling times.
# - One Numpy vector containing the time stamps for each joint 



class ProMP:
    def __init__(self, identifier=None,  TrainingData=None, params=None):
        self.identifier = identifier
        self.D=params['D'] #DoF
        self.K=params['K'] #Basis Functions
        self.N=params['N'] #Number of Demonstrations currently attached. 
        self.TrainingData=TrainingData
        

        

class robotoolbox: #several tools that come in handy for other scripts
    @staticmethod
    def GBasis(z,c,h): 
        """"Returns a gaussian basis"""
        
        return np.exp(-((z-c)**2)/(2*h))
    
        @staticmethod
    def PBasis(z,o): 
        """"Returns a polynomial basis of order o"""
        
        return z**o
    
    
    @staticmethod
    def why():
        whyv=npr.randint(1,5)
        if whyv==1:
            print("Because Claudia was too lazy to program it")
        if whyv==2:
            print("Because Sevin is busy making delicious baklavas")
        if whyv==3:
            print("Because Samuel went for a coffee two hours ago and hasn't returned")
        if whyv==4:
            print("Because they didn't order pizza for us")
    
   
    @staticmethod
    def IAmHungry():
        print('We were too lazy to hard-code a better asscii art cake. ')
        print("        iiiiiiiiiii \n       |___________| \n     __|___________|_")
    #Cake and more from: https://asciiart.website/index.php?art=events/birthday
        
Blob=ProMP(identifier='Blob', TrainingData=df, params=params)
  