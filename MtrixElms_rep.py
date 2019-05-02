import numpy as np
from matplotlib import cm
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.mlab import griddata
import pprint


N=100
l1=np.loadtxt('circular-light-matr-elem.dat')
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
newf=np.zeros((400,400))
for i in range(len(fx)):
	for j in range(len(fy)):
		newf[i,j]=f(0.5*fx[i]+0.288617*fy[j], -0.5*fx[i]+0.288617*fy[j])



#x=y=np.linspace(-1.1547,1.1547,400)
#f=interpolate.interp2d(x,y,Z,kind='cubic')
#Z=f(x,y)



#Z=f(x,y)
'''
func_0=(lambda x: x-N*np.floor(x/N))
f.x=f.x*N
f.y=f.y*N
f.x=func_0(f.x)
f.y=func_0(f.y)
'''


'''
#MATRIX ROTATION 
# To the left
size = len(Z)

for x in range(int(size/2)):
	for y in range(x,size-x-1):
		nx= size -1 -x
		ny= size -1-y
		swapVal=Z[x][y]
		Z[x][y]=Z[y][nx]
		Z[y][nx]=Z[nx][ny]
		Z[nx][ny]=Z[ny][x]
		Z[ny][x]=swapVal

# To the right
for x in range(int(size/2)):
	for y in range(x,size-x-1):
		nx= size -1 -x
		ny= size -1-y
		swapVal=Z[x][y]
		Z[x][y]=Z[ny][x]
		Z[ny][x]=Z[nx][ny]
		Z[nx][ny]=Z[y][nx]
		Z[y][nx]=swapVal

'''

x=np.linspace(-1*N,1*N,4*N)
y=np.linspace(-1*N,1*N,4*N)
f=interpolate.interp2d(x,y,newf,kind='cubic')



x=np.linspace(-1.1547,1.1547,4*N)
y=np.linspace(-1.1547,1.1547,4*N)
x=x*50
y=y*50
Z=f(x,y)
size=len(Z)
'''
for lx in range(int(size/2)):
	for ly in range(lx,size-lx-1):
		nx= size -1 -lx
		ny= size -1-ly
		swapVal=Z[lx][ly]
		Z[lx][ly]=Z[ly][nx]
		Z[ly][nx]=Z[nx][ny]
		Z[nx][ny]=Z[ny][lx]
		Z[ny][lx]=swapVal
'''
'''
for lx in range(int(size/2)):
	for ly in range(lx,size-lx-1):
		nx= size -1 -lx
		ny= size -1-ly
		swapVal=Z[lx][ly]
		Z[lx][ly]=Z[ny][lx]
		Z[ny][lx]=Z[nx][ny]
		Z[nx][ny]=Z[ly][nx]
		Z[ly][nx]=swapVal
'''
fig, ax=plt.subplots()

#im=ax.imshow(Z)
#print ax.transData

#im=ax.imshow(Z,interpolation='spline16',origin='lower',extent=[-1.1547,1.1547,-1.1547,1.1547],cmap='jet')
x=np.linspace(-1.1547,1.1547,4*N)
y=np.linspace(-1.1547,1.1547,4*N)
newZ=np.zeros((400,400))
new_x=[]
new_y=[]
for i in range(len(x)):
	if -1 < x[i] and x[i] < 1:
		for j in range(len(y)):
			if -1.1547 < y[j] and y[j] < 1.1547:
			
				if np.abs(y[j])<(-0.5773*np.abs(x[i])+1.1547):
			
					newZ[i,j]=Z[i,j]
					new_x.append(x[i])
					new_y.append(y[j])

#new_x=set(new_x)
#new_y=set(new_y)


newZ=np.ma.masked_where(newZ ==0,newZ)
newZ=np.ma.array(newZ)
#pprint.pprint(msk_Z)
#Z=Z.reshape((345,300))
im=ax.imshow(newZ,interpolation='spline16',origin='lower',extent=[-1.1547,1.1547,-1.1547,1.1547],cmap='jet')

''' 
#Plot with contour plot
Z=newZ

for lx in range(int(size/2)):
	for ly in range(lx,size-lx-1):
		nx= size -1 -lx
		ny= size -1-ly
		swapVal=Z[lx][ly]
		Z[lx][ly]=Z[ny][lx]
		Z[ny][lx]=Z[nx][ny]
		Z[nx][ny]=Z[ly][nx]
		Z[ly][nx]=swapVal
im=ax.contourf(x, y, Z, 8, alpha=.75, cmap='jet')
'''
#im=ax.pcolormesh(x, y, z)
transform=mtransforms.Affine2D().rotate_deg(90)
trans_data = transform + ax.transData

im.set_transform(trans_data)

plt.show()
