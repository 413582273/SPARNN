import numpy as np  
import math
from numpy.linalg import inv

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
from numpy import pi, cosh, exp, round, zeros, identity, arange, real, cos, sin, multiply, transpose
from numpy.fft import fft,ifft
from matplotlib.pyplot import figure
###########################################
'''
https://github.com/tanumoydhar/1D-Swift-Hohenberg-equation

Objective: To integrate 1d Swift-Hohenberg (SH) equation with periodic boundaries.

Method used: spectral method using Fourier modes in spatial dimension.
             time integration is performed using: second-order Adams-Bashforth + Crank-Nicholson 
			
			 (implicit treatment of linear terms and explicit treatment to non-linear terms) 

	The SH equation used-

	du/dt = ru - (qc^2 - (d/dx)^2)^2u + f(u)
	where, r = bifurcation parameter = 0.2 (we choose an arbitary value)

	qc = characteristic wavenumber = 1.0 (we choose q_c to be unity for infinite domain)

	f(u) = b_2*u^2	-u^3 (we choose the model SH23, where b_2 is a constant > 0) 

After substuting the respective values of constant the equation solved below appears as -

	du/dt = (r-1)u - (1 - 2*((d/dx)^2)) + (d/dx)^4))u + b_2*u^2 - u^3

Sample reference: 1d-Kuramoto-Sivashinksy (will be shortly uploaded https://github.com/PratikAghor
)
'''
###########################################
# Grid and required matrices:
Lx = 4*pi; #length of spatial domain along x-direction
nsave = 10;
Nt = 5000;
theta = 0.5; #weight to the current time-step value for the linear operator. theta = 0 => implicit, theta = 1 =>explicit, theta = 0.5 => Crank-Nicholson 

N_fm = 256; #no. of fourier modes
dt = 0.001;  #default time step

x = (1.0*Lx/N_fm)*arange(0, N_fm);
u = cos(x)# initial condition

kx = zeros(N_fm);
kx[0:int(N_fm/2)] = arange(0,N_fm/2); 
kx[int(N_fm/2)+1:] = arange(-N_fm/2+1,0,1);
alpha = 2.0*pi*kx/Lx;              # real wavenumbers:    exp(alpha*x)    
 
D = (1j*alpha); #* is element wise multiplication and D = d/dx operator in Fourier space

r = 0.1; #bifurcation parameter 
b_2 = 0.1;
bf_c = (r - 1.0);


L = (bf_c + 2.0*alpha*alpha - alpha*alpha*alpha*alpha);   # linear operator -D^2 - D^4 in Fourier space, diagonal matrix!

Nsave = int(Nt/nsave + 1);                           # number of saved time steps, including t=0
t = zeros(Nsave);
t[0:Nsave] = arange(0, Nsave)*(dt*nsave);       # t timesteps


U = zeros((Nsave, N_fm));                       # matrix of (ti, xj) values.
U[0,:] = u;                                     # assign initial condition to U
counter = 2 ;                                   # counter for saved data


# convenience variables
dt2  = 0.5*dt;
dt32 = 1.5*dt;

#linear algebra problem with Ax = B
A     = (np.ones(N_fm) - (1.0-theta)*dt*L)
A_inv = (np.ones(N_fm) - (1.0-theta)*dt*L)**(-1); # A inverse

B     = np.ones(N_fm) + theta*dt*L;

N1  =b_2*fft(u*u) + (-1.0)*fft(u*u*u); 
N0 = N1;                               
v  = fft(u);                           

#time stepping
for n in range (1, Nt):
    N0 = N1;
    N1 = b_2*fft((real(ifft( v )))*(real(ifft( v ))))  + (-1.0)*fft((real(ifft( v )))*(real(ifft( v )))*(real(ifft( v )))); 

    v = A_inv*((B*v) + dt32*N1 - dt2*N0);
    
    if n % nsave == 0:
        U[counter, :]  = real(ifft(v))
        counter = counter + 1

print(np.shape(U))
#########################################################################
#Import plotting functions:
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pylab 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse, Polygon
import numpy as np
import math
import pylab

font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : 25}
#matplotlib.rc('font', **font)
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Lucida Grande']})
rc('text', usetex=True)
mpl.rcParams.update({'font.size': 25})
from matplotlib import cm

fig1 = plt.figure()
ax1 = fig1.gca()

plt.contourf(t,x,np.transpose(U))
plt.xlabel("$t$")
plt.ylabel("$x$")
cbar = plt.colorbar()

plt.tight_layout()
plt.show()
fig1.savefig('spacetime_plot.eps')
################################################################

import scipy.io as scio
scio.savemat('sh_data.mat',{'uu':np.transpose(U)})