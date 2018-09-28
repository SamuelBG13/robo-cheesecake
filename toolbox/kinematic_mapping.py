import time
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

# this is a toolbox for transforming joint_states of the Fraka robot
# to cartesian coordinats using hartenberg-denavit transformation


def init_params():
    A=np.array([ 0,0,0,0.0825,-0.0825,0,0.088,0])
    D=np.array([ 0.333,0,0.316,0,0.384,0,0,0.107])
    Alph=(np.pi/2)*(np.array([0,-1,1,1,-1,1,1,0]))
    Thet=np.zeros(8)
    d = {'a':A, 'd':D,'alpha':Alph, 'theta': Thet}
    df = pd.DataFrame(d)

    return df


def Rx(alpha):
    ''' rotations around x axis'''
    transform=np.eye(4)

    transform[1,1]=np.cos(alpha)
    transform[2,2]=np.cos(alpha)
    transform[2,1]=np.sin(alpha)
    transform[1,2]=-np.sin(alpha)

    return transform


def Rz(theta):
    ''' rotations around z axis'''
    transform=np.eye(4)

    transform[0,0]=np.cos(theta)
    transform[1,1]=np.cos(theta)
    transform[1,0]=np.sin(theta)
    transform[0,1]=-np.sin(theta)

    return transform


def Transl(x,y, z):
    ''' shift in xyz-space'''
    transform=np.eye(4)

    transform[0,3]=x
    transform[1,3]=y
    transform[2,3]=z

    return transform


def HDtransfromation(params):
    ''' Complete transformation: (i) shift in z-direction (ii) rotation around z-axis (iii) shift in x-direction
	(iv) rotation around x-axis '''

    d = params['d']
    theta = params['theta']
    a = params['a']
    alpha = params['alpha']

    return np.dot(Rx(alpha), np.dot(Transl(a, 0, 0), np.dot(Rz(theta), Transl(0,0,d))))


def Transform2Base(df):
    ''' return list of seven matrices, each of which is the transformation of the i-th joint's coordinate system
	back to the base'''

    single_trans_list = [HDtransfromation(dict(df.loc[i])) for i in range(len(df))] # transformmation matrices from i-th to (i-1)-th coordinate system

    total = []
    total.append(single_trans_list[0])
    for i in range(1, len(single_trans_list)):
        total.append(np.dot(total[i-1], single_trans_list[i]))
    return total

def ExecuteTransform(matrix, vector):
    return np.round(np.dot(matrix , vector), 3)


def Zbounds(a):
    minZ=0.02
    maxZ=1.08
    return (a>minZ and a<maxZ)

def Xbounds(a):
    minX=0.28
    maxX=0.82
    return (a>minX and a<maxX)

def Ybounds(a):
    minY=-0.78
    maxY=0.78
    return (a>minY and a<maxY)

def Bounds(a, coord):
    if coord == 'X':
        return Xbounds(a)
    elif coord == 'Y':
        return Ybounds(a)
    elif coord == 'Z':
        return Zbounds(a)
    else:
        print("ERROR")
        return 0


def IsBetweenBounds(theta, df=None):
    if df is None:
        df=init_params()

    """This algorithm checks:

    1. That the robot stump doesn't hit the table during joint space explorations.
    2.  Guarantees that the robot samples trajectories within the safe box."""

    df['theta']=theta
    Matrices=Transform2Base(df)


    standard_jointcoordinateaxis=np.array((0,0,0,1)) #with repsect to all the joints
    """The most likely part of the robot that can crash is the small stump that has a keypad and a light, which is
    part of joint six, and is measured (and always constant with respect to) coordinate axis #6. In the next two lines,
    we measure the position of the two bounds of this stump """
    keypad_jointcoordinateaxis_border1=np.array((0.10,0.09,0.05,1)) #The keypad position With respect to the sixth coordinate axis
    keypad_jointcoordinateaxis_border2=np.array((0.10,0.09,-0.05,1))
    endef_jointcoordinateaxis_center=np.array((0,0,0.13,1)) #With respect to the last coordinate axi


    criticalpoint1=ExecuteTransform(Matrices[5],standard_jointcoordinateaxis)
    criticalpoint2=ExecuteTransform(Matrices[5],keypad_jointcoordinateaxis_border1)
    criticalpoint3=ExecuteTransform(Matrices[5],keypad_jointcoordinateaxis_border2)
    criticalpoint4=ExecuteTransform(Matrices[7],endef_jointcoordinateaxis_center)



    safe=True #InitialAssumption

    for coordinate, axis in enumerate(['X', 'Y', 'Z']):
        CPs=np.array([criticalpoint1[coordinate], criticalpoint2[coordinate], criticalpoint3[coordinate], criticalpoint4[coordinate]]) # A vector of the coordinates of our points of interest
        for cp in CPs:

            if not Bounds(cp, axis):
                safe =False
                break
        if safe==False:
            break

    return safe
