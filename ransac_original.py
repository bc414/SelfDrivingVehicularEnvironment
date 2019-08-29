import os
import random
import math
import argparse
import numpy as np
from numpy.random import normal, shuffle
import numpy.linalg

import scipy.optimize

def initFit(v_points):
    A=np.hstack((v_points*2, np.ones((len(v_points), 1))))
    f=np.sum(v_points**2, axis=1).reshape((len(v_points), 1))
    C, residuals, rank, singval=np.linalg.lstsq(A, f)
    t=C[3]+C[0]**2+C[1]**2+C[2]**2
    r=math.sqrt(t)
    return r, C[0], C[1], C[2]

def residual(variables, x):
    radius=variables[0]
    t_x=variables[1]
    t_y=variables[2]
    t_z=variables[3]
    return np.sqrt((x[:, 0]-t_x)**2+(x[:, 1]-t_y)**2+(x[:, 2]-t_z)**2)-radius

def fitSphere(v_points, v=False):
    r_init, t_xi, t_yi, t_zi=initFit(v_points)
    if v:
        print("Initial linear least square fit results:")
        print("\t Radius: {0}".format(r_init))
        print("\t Origin: {0}, {1}, {2}".format(t_xi, t_yi, t_zi))
    variables=[r_init, t_xi.item(), t_yi.item(), t_zi.item()]
    output=scipy.optimize.leastsq(residual, variables, args=v_points)
    r_o=output[0][0]
    tx_o=output[0][1]
    ty_o=output[0][2]
    tz_o=output[0][3]
    if v:
        print("Final non-linear least square fit results:")
        print("\t Radius: {0}".format(r_o))
        print("\t Origin: {0}, {1}, {2}".format(tx_o, ty_o, tz_o))
    return r_o, tx_o, ty_o, tz_o

def calc_sphere(vrs, theta_points, phi_points):
    g_pts=np.zeros((theta_points*phi_points, 3))
    idx=0
    for t_step in range(0, theta_points):
        for p_step in range(0, phi_points):
            theta=t_step*2*math.pi/theta_points
            phi=p_step*math.pi/phi_points
            g_pts[idx, :]=[vrs[1]+vrs[0]*math.cos(theta)*math.sin(phi), vrs[2]+vrs[0]*math.sin(theta)*math.sin(phi), vrs[3]+vrs[0]*math.cos(phi)]
            idx=idx+1
    return g_pts

def ransac(points, min_percentage, per_point_err, max_iter, thresh_percentage):
    numpoints=len(points)
    min_points=int(min_percentage*numpoints) if min_percentage < 1 else min_percentage
    thresh_points=int(thresh_percentage*numpoints)
    print(numpoints, min_points, thresh_points)
    bestErr=float('inf')
    bestInliers=[]
    r, tx, ty, tz=0, 0, 0, 0
    itr=0
    #while itr<max_iter:]
    while itr<max_iter:
        #print("Iteration {0}:".format(itr))
        random_indices=np.random.choice(numpoints, min_points, replace=False)
        r_m, tx_m, ty_m, tz_m=fitSphere(points[random_indices], v=False)
        test_indices=np.array([val for val in np.arange(0, numpoints) if val not in random_indices])
        res=residual([r_m, tx_m, ty_m, tz_m], points[test_indices])
        inliers=np.where(abs(res)<per_point_err)[0]
        #print("Found {0} inliers".format(len(inliers)))
        if len(inliers)>thresh_points:
            #r_b, tx_b, ty_b, tz_b=fitSphere(np.vstack((points[inliers], points[random_indices])), v=False)
            r_b, tx_b, ty_b, tz_b=fitSphere(points[inliers], v=False)
            #model_residuals=residual([r_b, tx_b, ty_b, tz_b], np.vstack((points[inliers], points[random_indices])))
            model_residuals=residual([r_b, tx_b, ty_b, tz_b], points[inliers])
            totalError=np.sum(model_residuals**2)
            if totalError<bestErr:
                print("!")
                bestErr=totalError
                bestInliers=inliers
                r, tx, ty, tz=r_b, tx_b, ty_b, tz_b
        itr=itr+1
        #kprint("\n")
    return ([r, tx, ty, tz], bestInliers)


def noise():
    return normal(0, .05)

parser=argparse.ArgumentParser(description='Generate noisy data on a sphere with outliers and attempt to fit model of sphere using RANSAC')
parser.add_argument('num_points', type=int, help="number of points generated")
parser.add_argument('outlier_ratio', type=float, help="number of outliers to generate as a multiple of num_points")
parser.add_argument('min_model_points', type=int, help="number of points used to generate a model")
parser.add_argument('per_point_error', type=float, help="minimum allowable error for inliers")
parser.add_argument('thresh_percentage', type=float, help="percentage of points which must be inliers for a model to be considered")
parser.add_argument('iterations', type=int, help="number of ransac iterations")

args=parser.parse_args()
print(args)

if os.path.exists("data"):
    os.remove("data")
f=open("data", "w")

outlier_ratio=args.outlier_ratio
numpoints=args.num_points
numoutliers=int(numpoints*outlier_ratio)

t_vec=np.matrix([[random.random()], [random.random()], [random.random()]])
R=3*random.random()+2
print("Ground truth:")
print("origin: ", t_vec[0].item(), t_vec[1].item(), t_vec[2].item())
print("radius: ", R)

m_pts=np.zeros((numpoints+numoutliers, 3))

theta=0
phi=math.pi/2

for i in range(0, numpoints):
    theta_p=theta+normal(0, .7)
    phi_p=phi+normal(0, .5)
    m_pts[i, :]=[t_vec[0].item()+R*math.cos(theta_p)*math.sin(phi_p)+noise(), t_vec[1].item()+R*math.sin(theta_p)*math.sin(phi_p)+noise(), t_vec[2].item()+R*math.cos(phi_p)+noise()]

for i in range(0, numoutliers):
    m_pts[i+numpoints, :]=[t_vec[0].item()+normal(0, R), t_vec[1].item()+normal(0, R), t_vec[2].item()+normal(0, R)]

shuffle(m_pts)
m_pts=np.load("noClipping.npy")
print("Shape:",m_pts.shape)
result=ransac(m_pts, args.min_model_points, args.per_point_error, args.iterations, args.thresh_percentage)
print("shape of result:",result.shape)
r, tx, ty, tz=result[0]
inliers=result[1]
outliers=[val for val in range(0, numpoints+numoutliers) if val not in inliers]
print("Shape of outliers:",len(outliers))
#print(outliers)
print("Solution:")
print("origin: ", tx, ty, tz)
print("radius: ", r)

if r>76 or r<70:
    print("Failed to find a solution")

g_pts=calc_sphere([r, tx, ty, tz], 30, 15)

for i in range(0, numpoints):
    f.write(str(m_pts[i][0])+"\t"+str(m_pts[i][1])+"\t"+str(m_pts[i][2])+"\n")

f.write("\n")
for i in range(0, len(g_pts)):
    f.write(str(g_pts[i][0])+"\t"+str(g_pts[i][1])+"\t"+str(g_pts[i][2])+"\n")

f.write("\n")
for i in range(0, len(outliers)):
    f.write(str(m_pts[outliers[i]][0])+"\t"+str(m_pts[outliers[i]][1])+"\t"+str(m_pts[outliers[i]][2])+"\n")

f.write("\n")
for i in range(0, len(inliers)):
    f.write(str(m_pts[inliers[i]][0])+"\t"+str(m_pts[inliers[i]][1])+"\t"+str(m_pts[inliers[i]][2])+"\n")

f.close()
