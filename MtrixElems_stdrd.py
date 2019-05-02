import numpy as np
from matplotlib import cm
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.mlab import griddata
import pprint


l1=np.loadtxt('circular-light-matr-elem.dat')
N=int(np.sqrt((len(l1)))-1)

lData=[]
for i in range(1,N+1):
	for j in range(1,N+1):
		er=l1[(i)*(N+1)+j,1]
		lData.append(er)
lData=np.array(lData)
lData=lData.reshape(N,N)
x=np.linspace(-1*N,1*N,4*N)
y=np.linspace(-1*N,1*N,4*N)

Z=lData
Z=np.append(Z,Z,axis=0)
Z=np.append(Z,Z,axis=0)
Z=np.append(Z,Z,axis=1)
Z=np.append(Z,Z,axis=1)

f=interpolate.interp2d(x,y,Z,kind='cubic')
Z=f(x,y)

fx=f.x
fy=f.y
newf=np.zeros((4*N,4*N))
for i in range(len(fx)):
	for j in range(len(fy)):
		newf[i,j]=f(fx[i], -0.5*fx[i]+0.866025*fy[j])



x=np.linspace(-1*N,1*N,4*N)
y=np.linspace(-1*N,1*N,4*N)
f=interpolate.interp2d(x,y,newf,kind='cubic')



x=np.linspace(-0.66667,0.66667,4*N)
y=np.linspace(-0.66667,0.66667,4*N)
x=x*50
y=y*50
Z=f(x,y)

fig, ax=plt.subplots()

#im=ax.imshow(Z,interpolation='spline16',origin='lower',extent=[-1.1547,1.1547,-1.1547,1.1547],cmap='jet')
x=np.linspace(-0.66667,0.66667,4*N)
y=np.linspace(-0.66667,0.66667,4*N)
newZ=np.zeros((4*N,4*N))
new_x=[]
new_y=[]
for i in range(len(x)):
	if -0.66667 < x[i] and x[i] < 0.66667:
		for j in range(len(y)):
			if -0.57735 < y[j] and y[j] < 0.57735:

				if np.abs(y[j])<(-1.73205*np.abs(x[i])+1.1547):

					newZ[j,i]=Z[i,j]


newZ=np.ma.masked_where(newZ ==0,newZ)
newZ=np.ma.array(newZ)
#pprint.pprint(msk_Z)
#Z=Z.reshape((345,300))

im=ax.imshow(newZ,interpolation='spline16',origin='upper',extent=[-1.1547,1.1547,-1.1547,1.1547],cmap='jet')

#im=ax.pcolormesh(x, y, z)
transform=mtransforms.Affine2D().rotate_deg(90)
trans_data = transform + ax.transData
#im=ax.contourf(x, y, Z, 8, alpha=.75, cmap='jet')

#im.set_transform(trans_data)

Y, X = np.ogrid[-1:1:500j, -1.141:1.141:500j]
ax.contour(X.ravel(),Y.ravel(),abs(Y)<=-1.73205*np.abs(X)+1.97,colors='black',linewidth=1.2,interpolation='spline')
ax.hlines(.9903,-0.5668,0.5668,linewidth=2.)
ax.hlines(-.9903,-0.5668,0.5668,linewidth=2.)

plt.axis("off")
plt.show()
