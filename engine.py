# -*- coding: utf-8 -*-
"""
engine.py:
    This script generates the engine for our ProMPs distro.
    The methods implemented were adapted for two papers by the labs of Jan Peters in Tübingen and Darmstadt.
    These papers are:
        
        
        [1]. Using Probabilistic Movement Primitives in Robotics 
        By Paraschos, Daniel, Peters and Neumann 
        https://www.ias.informatik.tu-darmstadt.de/uploads/Team/AlexandrosParaschos/promps_auro.pdf
        
        and
        
        [2]. Adaptation and Robust Learning of Probabilistic Movement Primitives
        By Gomez-Gonzalez, Neumann, Schelkopf and Peters
        https://arxiv.org/pdf/1808.10648.pdf
        
        Also there's input from
        
        [3]. Probabilistic Movement Primitives
        By Paraschos, Daniel, Peters and Neumann
        https://www.ias.informatik.tu-darmstadt.de/uploads/Publications/Paraschos_NIPS_2013a.pdf
        
    Autors of the code: 
        Cheescake team @ Deep Learning and Robotics Challenge 2018
        
"""
######################################################################################################

import numpy as np 
import numpy.random as npr
import numpy.linalg  as npl
import pickle as p 
import pandas as pd
import matplotlib.pyplot as plt     
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize, NonlinearConstraint,BFGS

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
        self.meanw=np.zeros(self.K*self.D ) #W is vector of dimension K*D. We initialize it here. 
        self.W=pd.DataFrame(columns=['W']) #This dataframe stores the already-trained w vectors.
        self.examplestrained=0 # This accounts to the number of rows of this dataframe
        self.estimate_m=None
        self.estimate_sd=None
    
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
        We use K gaussian basis functions-"""
        
        K=self.K
        BaVe=np.zeros(K)
        h=0.1*1/(K-1) # the width of the basis functions is selected to consistently divide the entire phase interval
        Interval=[-2*h, 1+2*h] #The interval between the first and the last basis functions.
        dist_int=(Interval[1]+abs(Interval[0]))/(K-1) #Effective distance between each basis
        c=Interval[0]
        for k in range (K):
            BaVe[k]=robotoolbox.GBasis(z,c,h)
            c=c+dist_int

        #return BaVe
        return BaVe/np.sum(BaVe) # normalization of the basis functions is suggested in [1]
        

        
    def BasisMatrix(self,z):
        """Generates the basis matrics for the joints at the phase level z.
        This DxKD matrix is evaluated at a certain phase Z=z (only one phase point!). 
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
        
        return BaMa
        
    def BigFuckingMatrix(self, demonstration):
        """Generates the basis matrics 'PSI' for all the joints over all the phase stamps.
        This is a DZxKD, Z being the number of phase stamps in the demonstration.. 
        This matrix is esentially a block matrix that stacks in the diagonal
        The parameter 'demonstration' is a series, taken as a row from the training data
        dataframe."""
        Z=demonstration["Phases"].shape[0]
        D=self.D
        K=self.K
        BFM=np.zeros((Z*D,K*D))
        index_phase=0 #On each row within the same DoF, the Basis Vector should start on a different index. The first one starts at 0
        index_block=0 #On each DoF of the basis matrix, the Basis Vector should start on a different index. The first one starts at 0


        for d in range(D):
            for idz, z in enumerate(demonstration["Phases"]):
                BFM[index_phase,index_block:index_block+K]=self.BasisVector(z)#Computes the basis vector, which is assumed to be the same for each joint
                index_phase=index_phase+1
            index_block=index_block+K
        
        return BFM
        
                
    def PlotBasisFunctions(self):
        """Simply plots the basis functions currently used"""
        plt.figure()
        q=np.arange(-0.1,1.1,0.005)
        for Q in q:
            
            for idx, basis in enumerate(self.BasisVector(Q)):
                plt.plot(Q,basis, 'b*')
                plt.title('Basis Functions')
       
    def RegularizedLeastSquares(self, l=1e-12):
        
        """This uses ther regularized least squares method to train the ProMP.
        It is described on [1]. 
        
        l is the regularization parameter.
        """
        D=self.D
        K=self.K
        
        Qlist=['q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6'] #list of q strings 
        wmatrix=np.array([])
        df=self.TrainingData
        for index, row in df.iterrows():
            PSI=self.BigFuckingMatrix(row)
            Y=row[Qlist[0]]
            for q in range (1,D):
                Y=np.hstack((Y,row[Qlist[q]]))
            ######
            RegMat=l*np.eye(K*D)
            
            factor1=npl.inv((np.matmul(PSI.T,PSI)+RegMat))
            factor2=np.dot(PSI.T,Y)
            w=np.dot(factor1,factor2)
            self.examplestrained=self.examplestrained+1
            
            # The exemplified w's are stored as a dataframe. But first, they are
            # stacked on a matrix such that we can compute easily the covariance. 
            if index==0:
                wmatrix=w
            else:
                wmatrix=np.vstack((wmatrix,w)) 
                
            self.W=self.W.append({'W': w}, ignore_index=True)

              
        self.estimate_m=np.mean(self.W.values)
        self.estimate_sd=np.std(self.W.values)
        self.estimate_sigma=np.cov(wmatrix.T)

    def ExpectationMaximization(self, l=1e-12):
        
        """This uses EM wihtout a prior. 
        

        """
        D=self.D
        K=self.K
        
        Qlist=['q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6'] #list of q strings 
        wmatrix=np.array([])
        plt.figure()
        df=self.TrainingData
        for index, row in df.iterrows():
            PSI=self.BigFuckingMatrix(row)
            Y=row[Qlist[0]]
            for q in range (1,D):
                Y=np.hstack((Y,row[Qlist[q]]))
            ######
            RegMat=l*np.eye(K*D)
            
            factor1=npl.inv((np.matmul(PSI.T,PSI)+RegMat))
            factor2=np.dot(PSI.T,Y)
            w=np.dot(factor1,factor2)
            self.examplestrained=self.examplestrained+1
            
            # The exemplified w's are stored as a dataframe. But first, they are
            # stacked on a matrix such that we can compute easily the covariance. 
            if index==0:
                wmatrix=w
            else:
                wmatrix=np.vstack((wmatrix,w)) 
            plt.plot(w,'r') 
        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(n_components=2)
        gmm.fit(wmatrix)
        print(gmm.covariances_.shape)
        plt.plot(gmm.means_[0],'b', lw=5) 
        plt.plot(gmm.means_[1],'g', lw=5)
        self.MeanAndStdPredictionPlot_notOriginalMean(mean=gmm.means_[0], std=np.diagonal(gmm.covariances_[0])**0.5, factor=2)

        self.MeanAndStdPredictionPlot_notOriginalMean(mean=gmm.means_[1], std=np.diagonal(gmm.covariances_[1])**0.5, factor=2)



         
   
#        self.estimate_m=np.mean(self.W.values)
#        self.estimate_sd=np.std(self.W.values)
#        self.estimate_sigma=np.cov(wmatrix.T)
#
#                

    def GeneratePrediction(self, w=None, Z=None):
        ''' Predicts the mean trajectoy for a given w. 
        
        In case no w is given, the mean from all the demonstrations is plotted'''
        if w is None: #I.e., if the user has not provided anyt¿
            w=self.estimate_m

        if Z is None: #I.e., if the user has not provided anyt¿
            Z=np.arange(0,1,0.01)

        Qlist=['q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6'] #list of q strings 
        Y=np.array([])
        
        if isinstance(Z,int) or isinstance(Z,float) :
            Y=np.dot(self.BasisMatrix(Z),w)
        else:
            for z in Z:
                if z==0:
                    Y=np.dot(self.BasisMatrix(z),w)
                else:
                    Y=np.vstack((Y,np.dot(self.BasisMatrix(z),w)))
                   
        return Z, Y
        

    def GenerateDemoPlot(self, xvariable="Times", toy=False):
    
        """"Shows a plot of the demonstrations. Change xvariable for "Phases" in case
        you want to plot q vs z instead of q vs t.
        Boolean variable toy displays a label in the title of the plot"""
        
        df=self.TrainingData
        Qlist=['q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6'] #list of q strings 
        plt.figure() # The generated data for each joint
    
    
        for index, row in df.iterrows():
            

            for q in range(7):
                plt.subplot(1,7,q+1)
                
                plt.plot(row[xvariable], row[Qlist[q]],'r')
                plt.title(Qlist[q])
                plt.ylim(-np.pi,np.pi)
                #plt.xticks([0, 6], size = 14)
                if q>0:
                    plt.yticks([])
                else: 
                    plt.yticks([-np.pi, 0, np.pi], ['$-\pi$', '$0$', '$\pi$'], size = 14)
            if toy:
                plt.suptitle('Toy demonstrated data')
            else:
                plt.suptitle('Demonstrated data')


            plt.show()
          
    
    def MeanAndStdPredictionPlot(self, factor=1):
        ''' Predicts the mean trajectoy and std after training, and produces
        a nice plotplot of the mean +- std trajectories for each DoF. 
        The parameter factor multiplies the standard deviation'''

        Z=np.arange(0,1,0.05)
        Qlist=['q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6'] #list of q strings 
        Y=np.array([])
        for z in Z:
            if z==0:
                Y=np.dot(self.BasisMatrix(z),self.estimate_m)
            else:
                Y=np.vstack((Y,np.dot(self.BasisMatrix(z),self.estimate_m)))
        Ylower=np.array([])
        for z in Z:
            if z==0:
                Ylower=np.dot(self.BasisMatrix(z),self.estimate_m-factor*self.estimate_sd)
            else:
                Ylower=np.vstack((Ylower,np.dot(self.BasisMatrix(z),self.estimate_m-factor*self.estimate_sd)))
        Yupper=np.array([])
        for z in Z:
            if z==0:
                Yupper=np.dot(self.BasisMatrix(z),self.estimate_m+factor*self.estimate_sd)
            else:
                Yupper=np.vstack((Yupper,np.dot(self.BasisMatrix(z),self.estimate_m+factor*self.estimate_sd)))
  
        plt.figure()
        for idq, q in enumerate(Qlist):
            plt.subplot(1,7,idq+1)

            plt.fill_between(Z, Yupper[:,idq], Ylower[:,idq], alpha=0.6)
            plt.plot(Z,Y[:,idq],'k',LineWidth=1)
            plt.title(Qlist[idq])
            plt.ylim(-np.pi,np.pi)
        plt.suptitle('Mean + '+str(factor)+' std predictions')
        
    def MeanAndStdPredictionPlot_notOriginalMean(self, mean, std, factor=1):
        ''' Predicts the mean trajectoy and std after training, and produces
        a nice plotplot of the mean +- std trajectories for each DoF. 
        The parameter factor multiplies the standard deviation'''

        Z=np.arange(0,1,0.05)
        Qlist=['q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6'] #list of q strings 
        Y=np.array([])
        for z in Z:
            if z==0:
                Y=np.dot(self.BasisMatrix(z),mean)
            else:
                Y=np.vstack((Y,np.dot(self.BasisMatrix(z),mean)))
        Ylower=np.array([])
        for z in Z:
            if z==0:
                Ylower=np.dot(self.BasisMatrix(z),mean-factor*std)
            else:
                Ylower=np.vstack((Ylower,np.dot(self.BasisMatrix(z),mean-factor*std)))
        Yupper=np.array([])
        for z in Z:
            if z==0:
                Yupper=np.dot(self.BasisMatrix(z),mean+factor*std)
            else:
                Yupper=np.vstack((Yupper,np.dot(self.BasisMatrix(z),mean+factor*std)))
  
        plt.figure()
        for idq, q in enumerate(Qlist):
            plt.subplot(1,7,idq+1)

            plt.fill_between(Z, Yupper[:,idq], Ylower[:,idq], alpha=0.6)
            plt.plot(Z,Y[:,idq],'k',LineWidth=1)
            plt.title(Qlist[idq])
            plt.ylim(-np.pi,np.pi)
        plt.suptitle('Mean + '+str(factor)+' std predictions')        
    
    def ConditionedAndStdPredictionPlot(self, wconditioned, factor=1, Qtarget=None, Ztarget=None):
        ''' Produces a nice plot of a conditioned trajectory '''
        if Qtarget is not None and Ztarget is not None:
            PlotViaPoints=True
        else:
                
            PlotViaPoints=False
      
        Z=np.arange(0,1,0.01)
        Qlist=['q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6'] #list of q strings 
        Y=np.array([])
        Ycond=np.array([])

        for z in Z:
            if z==0:
                Y=np.dot(self.BasisMatrix(z),self.estimate_m)
                Ycond=np.dot(self.BasisMatrix(z),wconditioned)

            else:
                Y=np.vstack((Y,np.dot(self.BasisMatrix(z),self.estimate_m)))
                Ycond=np.vstack((Ycond,np.dot(self.BasisMatrix(z),wconditioned)))

        Ylower=np.array([])
        for z in Z:
            if z==0:
                Ylower=np.dot(self.BasisMatrix(z),self.estimate_m-factor*self.estimate_sd)
            else:
                Ylower=np.vstack((Ylower,np.dot(self.BasisMatrix(z),self.estimate_m-factor*self.estimate_sd)))
        Yupper=np.array([])
        for z in Z:
            if z==0:
                Yupper=np.dot(self.BasisMatrix(z),self.estimate_m+factor*self.estimate_sd)
            else:
                Yupper=np.vstack((Yupper,np.dot(self.BasisMatrix(z),self.estimate_m+factor*self.estimate_sd)))
  
        plt.figure()
        for idq, q in enumerate(Qlist):
            plt.subplot(1,7,idq+1)
            
            plt.fill_between(Z, Yupper[:,idq], Ylower[:,idq], alpha=0.6)
            plt.plot(Z,Y[:,idq],'k',LineWidth=2)
            plt.plot(Z,Ycond[:,idq],'r',LineWidth=2)
            plt.title(Qlist[idq], size =16)
            if idq>0:
                plt.yticks([])
            else: 
                plt.yticks([-np.pi, 0, np.pi], ['$-\pi$', '$0$', '$\pi$'], size = 14)
            plt.xticks([0, 1], size = 14)
            if idq==6:
                plt.legend(('Mean trajectory','Mean trajectory after conditioning to via points'), loc = 4, fontsize = 12)

            
            if PlotViaPoints:
                for idx, qtarg in enumerate(Qtarget):
                    plt.plot(Ztarget[idx], qtarg[idq],'r*')
            plt.ylim(-np.pi,np.pi)
        plt.suptitle('Mean + '+str(factor)+' standard deviations - Conditioned trajectory for N=' + str(self.N) + ' demonstrations', size = 16)
    

    def Condition_JointSpace(self,Qtarget, Ztarget):
        
        """Qtarget and Ztarget contain one or more target points.
        
        This script conditions the distribution and generates a nice plot over it.
        returns only the mean.
        T"""
        K=self.K
        D=self.D
        N=self.N
        Ey_des=0.001*np.eye(D) #Accuracy matrix
        Ew=self.estimate_sigma
        Muw=self.estimate_m

          
            
        for idxq, q in enumerate(Qtarget):
            z=Ztarget[idxq]
        
        
            Psi=self.BasisMatrix(z).T  # In [1] this matrix is defined as a DxKD matrix. 
            # However later it isregarded as KD*D. Therefore, the transpose is used. 
            
            #From [1]:
            L=Ew.dot(Psi).dot(npl.inv(Ey_des+Psi.T.dot(Ew).dot(Psi)))
            Muw=Muw+L.dot(q.T-Psi.T.dot(Muw))
            Ew=Ew-L.dot(Psi.T.dot(Ew))
            
        self.ConditionedAndStdPredictionPlot(wconditioned=Muw, Qtarget=Qtarget, Ztarget=Ztarget, factor=2)
        return Muw

    
#    def Condition_TaskSpace(self, viapoints):
#        pass
        
    def GetJointData(self, w=None, MaxTime=None, robotrate=0.5):
        """Gets the joint data in a robot-friendly way."""
        if w is None:
            w=self.estimate_m
        if MaxTime is None:
            MaxTime=robotrate*8
        SampledPoints=int(MaxTime/robotrate)
        Q=np.array([])
        T=np.array([])

        
        for timestamp in range(SampledPoints+1):
            phase=timestamp/SampledPoints
            prediction=[self.GeneratePrediction(w=w, Z=phase)[1]]

            if timestamp==0:
                Q=prediction
                
            else:
                Q=np.vstack((Q,prediction))
            
            T=np.append(T, robotrate*timestamp)
        
        return T, Q
    
    
    def GetStartPoint(self, w=None):
        if w is None:
            w=self.estimate_m
        
                  
        return self.GeneratePrediction(w=w, Z=0)[1]
    

    
        
            
       
        
            
                    

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
        whyv=npr.randint(1,11)
        if whyv==1:
            print("Because Claudia was too lazy to program it")
        if whyv==2:
            print("Because Sevin is busy making delicious baklavas")
        if whyv==3:
            print("Because Samuel went for a coffee two hours ago and hasn't returned")
        if whyv==4:
            print("Because they didn't order pizza for us")
        if whyv==5:
            print("Because Claudia  was watching Otto e Mezzo and she forgot she left the robot exploring alone")
        if whyv==6:
            print("Because Sevin forgot to import the weights from Caffe in TensorFlow")
        if whyv==7:
            print("Because Samuel's brain was shut down and the re-boots are not working")
        if whyv==8:
            print("I don't know, why do you ask me?")
        if whyv==9:
            print("El holandéeees voladoooor")
        if whyv==10:
            print("Because you failed to give me the answer for the question of life, the universe, and everything else")
    
    @staticmethod
    def GetAccel(Q,T):
        """ Q is a points x joints matrix. Qp and Qpp idem """
        D=Q.shape[1]
        Qp=np.zeros(Q.shape); Qpp=np.zeros(Q.shape); #Initializaiton.
        for joint in range(D):
            Qp[:,joint]=np.gradient(Q[:,joint], T)
            Qpp[:,joint]=np.gradient(Qp[:,joint], T)
        return Qp, Qpp
   
    @staticmethod
    
    def PrepareData(filename):
        """Loads the dataframe, which is in a pickle file, and puts it
        on the necessary shape to get in the ProMP class
        
        These are necessary preprocessing step """
        
        columns=['Times', 'q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6']
        df_generated = pd.DataFrame(columns=columns) #Empty dataframe
        
        df_origin = p.load(open(filename, 'rb'))
        N=df_origin['demo'].max()+1 #+1 because we labelled a demo as "0"
        for demo in range(N):
            temp=df_origin[df_origin['demo']==demo]
            timestamps=np.array(list(temp['timestamp_franka'].values))
            timestamps=timestamps-timestamps[0]
            q=np.array(list(temp['joint_pos'].values))
            df_generated=df_generated.append(pd.DataFrame(data={'Times': [timestamps], \
                                'q0': [q[:,0]], \
                                'q1': [q[:,1]], \
                                'q2': [q[:,2]], \
                                'q3': [q[:,3]], \
                                'q4': [q[:,4]], \
                                'q5': [q[:,5]], \
                                'q6': [q[:,6]]}),ignore_index=True)

            
        return df_generated, N
    

    @staticmethod
    def PlotTrajectory(T, Q):
        Qlist=['q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6'] #list of q strings 
        plt.figure() # The generated data for each joint
    
    
        for index, Time in enumerate(T):
            Traj=Q[index]
            for q in range(7):
                plt.subplot(1,7,q+1)
                
                plt.plot(Time, Traj[q],'g*')
                plt.title(Qlist[q])
                plt.ylim(-np.pi,np.pi)
            plt.suptitle('Oncoming trajectory')
        
    @staticmethod
    def PlotAccel(T, Qpp):
        Qlist=['q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6'] #list of q strings 
        plt.figure() # The generated data for each joint
    
    
        for index, Time in enumerate(T):
            Traj=Qpp[index]
            for q in range(7):
                plt.subplot(1,7,q+1)
                
                plt.plot(Time, Traj[q],'g*')
                plt.title(Qlist[q])
                plt.ylim(-np.pi,np.pi)
            plt.suptitle('Oncoming trajectory')        
    
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
                anglesstamp=baselineth+timestamp*variation+(((1+timestamp)**3)/80) #Gets angles numbers that increase with time
                anglesstamp= np.multiply(anglesstamp, np.array([-1,1,-1,1,-1,1,-1]))
                if i==0:
                    q=anglesstamp
                else:
                    q=np.vstack((q, anglesstamp))
            return q
        ##
            

        columns=['Times', 'q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6']
        df = pd.DataFrame(columns=columns) #Empty dataframe
        for n in range(N): #Creates N demonstrations, including the times vector and the angles  
            stampnum, Timevec=GenerateTimesVector(rate=50)
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
    
    
#N=8000
#params = {'D' : 7, 'K' :  8, 'N' : N}

#       
#Blob=ProMP(identifier='Blob', TrainingData=robotoolbox.GenerateToyData(N=N), params=params)
#robotoolbox.GenerateDemoPlot(Blob.TrainingData, xvariable='Phases')
#Blob.RegularizedLeastSquares() #Choice for l from [1]
#Ztarget=np.array([0.15,0.9])
#Qtarget=np.array([[0.08,-0.08,0.08,-1.7,0,1.75,0],[-0.3,0.3,-1,-0.6,-1.1,2.8,-0.75]])
#Blob.Condition_JointSpace(Qtarget, Ztarget)
        
#%%  
from toolbox import kinematic_mapping as km

class FrankaPanda:          
    """Forward and Inverse kinematics function for our robot
    Nomenclature of coordinate frames
    Base=0= The frame on the base of the robot
    1....7... the joints of the robot
    F= The end of the robot without the gripper, and without a rottion, as seen here:
    https://frankaemika.github.io/docs/control_parameters.html
    EndEff=EE=The end effector, which is F rotated -45° in the Z axis, it's consequent with the camera mainly     
        
        """
    def Camera(Q, Point):
        """Point is an np array of [x, y, z] with ONLY ONE POINT
        x, y and z MUST BE IN METERS """
        df=km.init_params()
        df['theta'].values[0:7]=Q
        TF_Base=km.Transform2Base(df)[7]
        TEE_F=km.Rz((-3/4)*np.pi) #Fixed
        TEE_Base=np.dot(TF_Base,TEE_F)
        PCamera_EE=np.array([0,0.06,0.08, 1])

        PPoint_EE=Point+PCamera_EE[0:3]
        PPoint_EE=np.append(PPoint_EE,1)
        print("End effector positioN", km.ExecuteTransform(TEE_Base, np.array([0,0,0,1])))
    
        return km.ExecuteTransform(TEE_Base, PPoint_EE)
    
        
    def fk(Q):
        df=km.init_params()
        df['theta'].values[0:7]=Q
        TF_Base=km.Transform2Base(df)[7]
 
        PGripper_F=np.array([0,0,0.11,1])
        PGripper_Base=km.ExecuteTransform(TF_Base, PGripper_F)
        return PGripper_Base[0:3]
    
    def fk_diffSystem(Q, system='shoulder'):
        if system=='shoulder':
            s=4
        elif system=='head':
            s=6
        else:
            s=7
        df=km.init_params()
        df['theta'].values[0:7]=Q
        TF_Base=km.Transform2Base(df)[s]
 
        PSystem_F=np.array([0,0,0,1])
        PSystem_Base=km.ExecuteTransform(TF_Base,  PSystem_F)
        return PSystem_Base[0:3]
    
    def LossProMP(q, ProMP, z):
        
        Z, Q_proMP=ProMP.GeneratePrediction(Z=z)
        return mean_squared_error(Q_proMP, q)
     
    def LossIK(q, Target, system=None):
        if system is None or system=='Gripper':
            loss= mean_squared_error(FrankaPanda.fk(q), Target)
        else:
            loss= mean_squared_error(FrankaPanda.fk_diffSystem(Q, system=system), Target)
         
            
        
        return loss 

    
    def ik(z, Target, ProMP):    
        res = minimize(FrankaPanda.LossIK, args=(Target), x0=ProMP.GeneratePrediction(Z=z)[1], method='COBYLA', tol=1e-3)
        print("IK error: ", FrankaPanda.LossIK(res.x, Target) ,"\nThe angles are: ", res.x, "\nThe position is: ", FrankaPanda.fk(res.x), "\nThe error is: ", (FrankaPanda.fk(res.x)-Target),  "\nThe differences with the ProMP are: ", (res.x-ProMP.GeneratePrediction(Z=z)[1]))
        return res.x

        
#%%
Q=np.array([0,0,0,-np.pi/2,0,np.pi/2, 0])       
Point=np.array([-0.1,-0.1, 0.6]) 
#print(FrankaPanda.Camera(Q, Point))

#%%    
"""
*-*-*-*-*-*-*-*-*-*-*-**-*-*-
"""

def main():
    
    #
    df_generated, N=robotoolbox.PrepareData('JointsFinalPresentation_take2.p')
    
    params = {'D' : 7, 'K' : 5, 'N' : N}
    RobotSaysHi=ProMP(identifier='RobotSaysHi', TrainingData=df_generated, params=params)
    RobotSaysHi.PlotBasisFunctions()
    RobotSaysHi.RegularizedLeastSquares() #Choice for l from [1]
    RobotSaysHi.GenerateDemoPlot(xvariable="Times")
#    
    z=1
    print(FrankaPanda.fk(RobotSaysHi.GeneratePrediction(Z=z)[1]))

    Target=np.array([ 0.711, -0.317,  0.106])
    Q_des=FrankaPanda.ik(z, Target, RobotSaysHi)
    w_des=RobotSaysHi.Condition_JointSpace(Qtarget=[RobotSaysHi.GetStartPoint(w=RobotSaysHi.estimate_m), Q_des], Ztarget=[0,z])        
#    
#    
    T, Q= RobotSaysHi.GetJointData(w=w_des)
#    
    Qp,Qpp=robotoolbox.GetAccel(Q,T)

    RobotSaysHi.ExpectationMaximization()
    
    
main()

