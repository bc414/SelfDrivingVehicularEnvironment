import os
import random
import math
import numpy as np

if os.path.exists("data"):
    os.remove("data")

f=open("data", "w")

sample_points=60

def noise():
    return random.gauss(0, .1)

def rotmat_x(angle):
    return np.matrix([[1, 0, 0],[0, math.cos(angle), -math.sin(angle)],[0, math.sin(angle), math.cos(angle)]])

def rotmat_y(angle):
    return np.matrix([[math.cos(angle), 0, math.sin(angle)],[0, 1, 0],[-math.sin(angle), 0, math.cos(angle)]])

def rotmat_z(angle):
    return np.matrix([[math.cos(angle), -math.sin(angle), 0],[math.sin(angle), math.cos(angle), 0],[0, 0, 1]])

x_angle=random.uniform(0, math.pi)
z_angle=random.uniform(0, 2*math.pi)

rot_mat=rotmat_z(z_angle)*rotmat_x(x_angle)
t_vec=np.matrix([[random.random()], [random.random()], [random.random()]])

pts_o=[]
pts_t=[]

#~~~~~~~~~~~~~~~~Generate Points~~~~~~~~~~~~~~~~~~~~~~
for i in range(0, sample_points):
    u=random.uniform(-math.pi, math.pi)
    v=random.uniform(-math.pi, math.pi)
    xo=2*math.cos(u)+.2*math.cos(u)*math.cos(v)
    yo=2*math.sin(u)+.2*math.sin(u)*math.cos(v)
    zo=.2*math.sin(v)
    po=np.matrix([[xo], [yo], [zo]])
    pt=rot_mat*po+t_vec+np.matrix([[noise()], [noise()], [noise()]])
    pts_o.append(po)
    pts_t.append(pt)

def calcMatrix (pts_o,pts_t):
#~~~~~~~~~~~~~~~~~~Write Points to File~~~~~~~~~~~~~~~~~~~~~
	for i in range(0, len(pts_o)):
	    f.write(str(pts_o[i].item(0))+"\t"+str(pts_o[i].item(1))+"\t"+str(pts_o[i].item(2))+"\n")

	f.write("\n")

	for i in range(0, len(pts_t)):
	    f.write(str(pts_t[i].item(0))+"\t"+str(pts_t[i].item(1))+"\t"+str(pts_t[i].item(2))+"\n")

	#~~~~~~~~~~~~~~~~Calculate Centroids~~~~~~~~~~~~~~~~~~~~~~
	o_centroid=np.matrix([[0], [0], [0]], dtype=float)
	t_centroid=np.matrix([[0], [0], [0]], dtype=float)

	for p in pts_o:
	    o_centroid=o_centroid+p
	o_centroid=o_centroid/len(pts_o)

	for p in pts_t:
	    t_centroid=t_centroid+p
	t_centroid=t_centroid/len(pts_t)

	#~~~~~~~~~~~~~~~~~~~~~Calculate H~~~~~~~~~~~~~~~~~~~~~~~~
	H=np.matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=float)

	for o, t in zip(pts_o, pts_t):
	    H=H+(o-o_centroid)*(t-t_centroid).T

	u, s, vh=np.linalg.svd(H, full_matrices=True)

	rot_est=vh.T*u.T
	t_est=t_centroid-rot_est*o_centroid
	np.save('rotationMatrix',rot_mat-rot_est)
	np.save('translationVector',t_vec-t_est)	
	print(np.linalg.det(rot_est))
	print(rot_mat-rot_est)
	print(t_vec-t_est)
	#~~~~~~~~~~~~~~Generate Verification Points~~~~~~~~~~~~~~~~

f.write("\n")
error_sum=0
for i in range(0, len(pts_o)):
    pts_v=rot_est*pts_o[i]+t_est
    f.write(str(pts_v.item(0))+"\t"+str(pts_v.item(1))+"\t"+str(pts_v.item(2))+"\n")
    error_sum=error_sum+np.linalg.norm(pts_t[i]-pts_v)

print(error_sum/sample_points)
