# -*- coding: utf-8 -*-
"""
engine.py:
    This script generates the engine for our ProMPs distro.
    The methods implemented were adapted for two papers by the labs of Jan Peters in Tübingen and Darmstadt.
    These papers are:
        
        
        1. Using Probabilistic Movement Primitives in Robotics 
        By Paraschos, Daniel, Peters and Neumann 
        https://www.ias.informatik.tu-darmstadt.de/uploads/Team/AlexandrosParaschos/promps_auro.pdf
        
        and
        
        2. Adaptation and Robust Learning of Probabilistic Movement Primitives
        By Gomez-Gonzalez, Neumann, Schelkopf and Peters
        https://arxiv.org/pdf/1808.10648.pdf
        
    Autors of the code: 
        Cheescake team @ Deep Learning and Robotics Challenge 2018
        
"""
######################################################################################################

import numpy as np 
import numpy.random as npr
import pandas as pd
import matplotlib.pyplot as plt          
plt.close("all") 
           
    

   
#%%

"""
# ProMP is the main class of the engine.  
# The constructor method requires two inputs:
# - A string Identifier which roughly describes the task in hand
# - A dataframe TrainingData which contains all the examples for the robot. 
# - A dictionary containing parameters: DoF number, number of basis functions, number of demonstrations  → (D,K,N)
# !!WICHTIG: The number of basis functions should be > 2, since it uses 1 polynomial function and >1 
# 
# The dataframe should include one row for each demonstration, which should have: 
# - A NumPy vector of sampling times.
# - One Numpy vector containing the time stamps for each joint 

"""

class ProMP:
    def __init__(self, identifier=None,  TrainingData=None, params=None):
        self.identifier = identifier
        self.D=params['D'] #Number of degrees of freedom
        self.K=params['K'] #Number of basis Functions
        self.N=params['N'] #Number of Demonstrations currently attached. 
        self.TrainingData=TrainingData
        self.PhaseClipping() #At start adds the phase vectors
        self.w=np.zeros(self.K*self.D ) #W is matrix with K rows and D columns. 
        self.BasisMatrix(0.5)
        self.PlotBasisFunctions()
    
    def PhaseClipping(self):
        """Adds phase vectors to the the training dataframe. 
        The result is that all the demonstrations are scaled such that they
        last between 0 and 1 in phase space.
        
        This function assumes that all the demonstrations have a time vector
        which already starts at 0"""
        
        df = pd.DataFrame(columns=['Phases']) #Empty dataframe

        for index, row in self.TrainingData.iterrows():
            Z=row["Times"]/np.max(row["Times"])
            df=df.append(pd.DataFrame(data={'Phases': [Z]}), ignore_index=True)
        self.TrainingData=self.TrainingData.join(df)
    
    
    def BasisVector(self, z):
        """Generates the K x 1 basis vector for the ProMP. 
        We use K-1 gaussian basis functions and a first order polynomial,
        as reported by Gomez-Gonzales et. al. to produce best results."""
        
        K=self.K
        BaVe=np.zeros(K)
        h=0.01*1/(K-1) # the width of the basis functions is selected to consistently divide the entire phase interval
        Interval=[-2*h, 1+2*h] #The interval between the first and the last basis functions.
        dist_int=(Interval[1]+abs(Interval[0]))/(K-2) #Effective distance between each basis
        c=Interval[0]
        for k in range (K-1):
            BaVe[k]=robotoolbox.GBasis(z,c,h)
            c=c+dist_int
        BaVe[K-1]=z #Polynomial, first order
        return BaVe

    def BasisMatrix(self,z):
        """Generates the basis matrics for the joints at the phase level z.
        This DxKD matrix is evaluated at a certain phase z. 
        This matrix is esentially a block matrix that stacks in the diagonal
        the basis vectors for each degree of fredom.
        The basis vectors are, for simplicity, the same for each DoF and will be
        described first. """
        D=self.D
        K=self.K
        BaMa=np.zeros((D,K*D))

        
        index_block=0 #On each row of the basis matrix, the Basis Vector should start on a different index. The first one starts at 0
        for d in range(D):
            BaMa[d,index_block:index_block+K]=self.BasisVector(z)#Computes the basis vector, which is assumed to be the same for each joint
            index_block=index_block+K
        
        print(BaMa)
        
    def PlotBasisFunctions(self):
        """Simply plots the basis functions currently used"""
        plt.figure()
        q=np.arange(-0.1,1.1,0.005)
        for Q in q:
            
            for idx, basis in enumerate(self.BasisVector(Q)):
                plt.plot(Q,basis, 'b*')
                plt.title('Basis Functions')
       
         
    

        

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
        """Wait, but why?"""
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
    def GenerateDemoPlot(df, xvariable="Times"):
        """"Shows a plot of the demonstrations. Change xvariable for "Phases" in case
        you want to plot q vs z instead of q vs t."""
        
        Qlist=['q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6'] #list of q strings 
        plt.figure() # The generated data for each joint
    
    
        for index, row in df.iterrows():
            for q in range(7):
                plt.subplot(1,7,q+1)
                
                plt.plot(row[xvariable], row[Qlist[q]])
                plt.title(Qlist[q])
                plt.ylim(-np.pi,np.pi)
                plt.suptitle('Toy demonstrated data')
                
                
   
    @staticmethod
    def IAmHungry():
        print('We were too lazy to hard-code a better asscii art cake. ')
        print("        iiiiiiiiiii \n       |___________| \n     __|___________|_")
    #Cake and more from: https://asciiart.website/index.php?art=events/birthday
    
    @staticmethod
    def GenerateToyData(N=6):
        """Generated a dataframe with toy data with N demonstrations. 
        This dataframe contains already vectors with the joint timestamps and vectors of time
        which necesarilly start a 0."""
        
        
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
    
    
N=3
params = {'D' : 7, 'K' : 4, 'N' : N}
       
Blob=ProMP(identifier='Blob', TrainingData=robotoolbox.GenerateToyData(N=N), params=params)
robotoolbox.GenerateDemoPlot(Blob.TrainingData, xvariable='Phases')


