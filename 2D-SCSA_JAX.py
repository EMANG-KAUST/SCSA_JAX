#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 14:39:47 2023

@author: vargasjm
"""

import numpy as jnp
from scipy.sparse import spdiags
from skimage import io, data
import matplotlib.pyplot as plt
import math
import numpy.matlib
from skimage import data
from skimage.metrics import structural_similarity, peak_signal_noise_ratio,mean_squared_error 

import cProfile as profile
import pstats
import jax
import jax.numpy as jnp
import jax.numpy.linalg as jla


def repmat(array, m, n):
    # Get the shape of the original array
    shape = array.shape

    # Expand the shape based on the desired repetitions
    new_shape = (shape[0] * m, shape[1] * n)

    # Reshape the original array to a higher-dimensional array
    expanded_array = array.reshape((shape[0], 1, shape[1], 1))

    # Tile the array along the specified dimensions
    tiled_array = jnp.tile(expanded_array, (1, m, 1, n))

    # Reshape the tiled array to the desired shape
    tiled_array = tiled_array.reshape(new_shape)

    return tiled_array
# np.warnings.filterwarnings('ignore', category=jnp.VisibleDeprecationWarning)                 

def SCSA2D(img,h,gm,fe):

    V=0.5*img;
    n=V.shape[0]
    psinnory=jnp.zeros([n,n,n],dtype='float32')
    psinnorx=jnp.zeros([n,n,n],dtype='float32')
    kappay=jnp.zeros([n,n],dtype='float32')
    kappax=jnp.zeros([n,n],dtype='float32')
    Nyv=jnp.zeros([n,1],dtype='float32')
    Nxv=jnp.zeros([n,1],dtype='float32')
    Ny_v=jnp.zeros([n,1],dtype='float32')
    Nx_v=jnp.zeros([n,1],dtype='float32')
    lamda=jnp.zeros([n,n],dtype='float32')
    psiy=jnp.zeros([n,n],dtype='float32')
    temp=jnp.zeros([n,1],dtype='float32')
    V1=jnp.zeros([n,n],dtype='float32')
    V2=jnp.zeros([n,n],dtype='float32')
    
    feh=2*jnp.pi/n;
    L=(1/(4*jnp.pi))/(gm+1)
    h2L = h_min**2/L;
    Dx=delta(n,fe,feh)
    D=-h_min*h_min*Dx
    
    for q in range(n):
    
        SCy=D-jnp.diag(V[:,q])
        [lamda,psiy]=jla.eigh(SCy)
        temp=lamda
        temp1=temp[temp<0]
        Ny=len(temp1)
        Ny_v=Ny_v.at[q].set(Ny)
        
        kappay=kappay.at[q,0:Ny].set(-temp1)
    
        
        psiny=psiy[:,0:Ny]**2   
        Iy=simp(psiny,fe);
        II=jnp.diag(1/Iy)
        Nyv=Nyv.at[q].set(Ny)
        
        psinnory=psinnory.at[q,:,0:Ny].set(jnp.matmul(psiny,II))
    
    
    
    
    
    
    for i in range(n):
      
        SCx=D-jnp.diag(V[i,:])
        [lamdax,psix]=jla.eigh(SCx)
        temp=lamdax
        temp1x=-temp[temp<0];
        Nx=len(temp1x);        
        Nx_v=Nx_v.at[i].set(Nx)
    
        kappax=kappax.at[i,0:Nx].set(temp1x)
        
        psinx=psix[:,0:Nx]**2;
        Iy=simp(psinx,fe);
        II=jnp.diag(1./Iy);
        psinnorx1=jnp.matmul(psinx,II)
        
        psinnorx=psinnorx.at[i,:,0:Nx].set(psinnorx1)
    
      
        for j in range(n):
            
            Kapx=repmat(temp1x.reshape(-1,1),1,int(Nyv[j]))
            Kapy=repmat(kappay[j,0:int(Nyv[j])].reshape(1,-1),Nx,1)
            Kap = Kapx+Kapy
            if gm == 4:
                Kap = Kap*(Kap)
                Kap = Kap*(Kap)
            elif gm == 3:
                Kap1 = Kap;
                Kap = Kap*(Kap)
                Kap = Kap*(Kap1)
            elif gm == 2:
                Kap = Kap*(Kap);
            else:
                Kap = Kap**(gm);
            
            mul1=jnp.matmul(psinnorx1[j,0:Nx].reshape(1,-1),Kap)
            
            mul=jnp.matmul(mul1,psinnory[j,i,0:int(Nyv[j])].reshape(int(Nyv[j]),1))
            
            V1=V1.at[i,j].set(h2L*mul[0][0])
    
        
        Nxv=Nxv.at[i].set(Nx)
            
    
    
    V2 = (V1**(1/(gm+1)));
    
    return V2,psinnory,psinnorx,V1,kappax,kappay,Nx_v,Ny_v

def SCSA2D2(img, h, gm, fe):
    V = 0.5 * img
    n = V.shape[0]

    psinnory = jnp.zeros([n, n, n], dtype='float32')
    psinnorx = jnp.zeros([n, n, n], dtype='float32')
    kappay = jnp.zeros([n, n], dtype='float32')
    kappax = jnp.zeros([n, n], dtype='float32')
    Nyv = jnp.zeros([n, 1], dtype='float32')
    Nxv = jnp.zeros([n, 1], dtype='float32')
    Ny_v = jnp.zeros([n, 1], dtype='float32')
    Nx_v = jnp.zeros([n, 1], dtype='float32')
    lamda = jnp.zeros([n, n], dtype='float32')
    psiy = jnp.zeros([n, n], dtype='float32')
    temp = jnp.zeros([n, 1], dtype='float32')
    V1 = jnp.zeros([n, n], dtype='float32')
    V2 = jnp.zeros([n, n], dtype='float32')

    feh = 2 * jnp.pi / n
    L = (1 / (4 * jnp.pi)) / (gm + 1)
    h2L = h ** 2 / L
    Dx = delta(n, fe, feh)
    D = -h * h * Dx

    for q in range(n):
        SCy = D - jnp.diag(V[:, q])
        lamda, psiy = jnp.linalg.eigh(SCy)
        temp = lamda
        temp1 = temp[temp < 0]
        Ny = len(temp1)
        Ny_v = Ny_v.at[q].set(Ny)
        kappay = kappay.at[q, 0:Ny].set(-temp1)
        psiny = psiy[:, 0:Ny] ** 2
        Iy = simp(psiny, fe)
        II = jnp.diag(1 / Iy)
        Nyv = Nyv.at[q].set(Ny)
        psinnory = psinnory.at[q, :, 0:Ny].set(jnp.matmul(psiny, II))

    for i in range(n):
        SCx = D - jnp.diag(V[i, :])
        lamdax, psix = jnp.linalg.eigh(SCx)
        temp = lamdax
        temp1x = -temp[temp < 0]
        Nx = len(temp1x)
        Nx_v = Nx_v.at[i].set(Nx)
        kappax = kappax.at[i, 0:Nx].set(temp1x)
        psinx = psix[:, 0:Nx] ** 2
        Iy = simp(psinx, fe)
        II = jnp.diag(1. / Iy)
        psinnorx1 = jnp.matmul(psinx, II)
        psinnorx = psinnorx.at[i, :, 0:Nx].set(psinnorx1)

        for j in range(n):
            Kapx = jnp.tile(temp1x.reshape(-1, 1), (1, int(Nyv[j])))
            Kapy = jnp.tile(kappay[j, 0:int(Nyv[j])].reshape(1, -1), (Nx, 1))
            Kap = Kapx + Kapy

            if gm == 4:
                Kap = Kap * Kap
                Kap = Kap * Kap
            elif gm == 3:
                Kap1 = Kap
                Kap = Kap * Kap
                Kap = Kap * Kap1
            elif gm == 2:
                Kap = Kap * Kap
            else:
                Kap = Kap ** gm

            mul1 = jnp.matmul(psinnorx1[j, 0:Nx].reshape(1, -1), Kap)
            mul = jnp.matmul(mul1, psinnory[j, i, 0:int(Nyv[j])].reshape(int(Nyv[j]), 1))
            V1 = V1.at[i, j].set(h2L * mul[0][0])

        Nxv = Nxv.at[i].set(Nx)

    V2 = (V1 ** (1 / (gm + 1)))
    #Nh = jnp.floor(jnp.mean([jnp.transpose(Nyv), Nxv]))

    return V2, psinnory, psinnorx, V1, kappax, kappay, Nx_v, Ny_v
def delta(n,fe,feh):
    
    array_q=jnp.arange(1,n)
    array=array_q[::-1]
 #   array=array_i.reshape(1,-1)

    ex=jnp.kron(array,jnp.ones((n,1), dtype=int));
    if jnp.mod(n,2)==0:
        dx=-math.pi**2/(3*feh**2)-(1/6)*jnp.ones((n,1), dtype=int);
        
        test_bx=(-(-1)**ex*(0.5))/(jnp.sin(ex*feh*0.5)**2);
        test_tx=(-(-1)**ex*(0.5))/(jnp.sin((-ex)*feh*0.5)**2);
    else:
        dx=-math.pi**2/(3*feh**2)-(1/12)*jnp.ones((n,1), dtype=int);
        test_bx=-0.5*((-1)**ex)* (1/jnp.tan(ex*feh*0.5))/(jnp.sin(ex*feh*0.5));
        test_tx=-0.5*((float(-1))**(-ex))* (1/jnp.tan((-ex)*feh*0.5))/(jnp.sin((-ex)*feh*0.5));
    
    Ex=spdiags(jnp.transpose(jnp.concatenate([test_bx, dx, test_tx],axis=1)),jnp.concatenate([(-1*jnp.arange(0,n)[::-1]),array],axis=0),n,n).toarray()
    
    Dx=(feh/fe)**2*Ex
    
    return Dx



def simp(f,dx):


    n=len(f)
    
    if n>1:
   
        I=1/3*(f[1,:]+f[2,:])*dx;

        for i in range(3,n):
            if(jnp.mod(i,2)==0):
                I=I+(1/3*f[i,:]+1/3*f[i-1,:])*dx
            else:
                I=I+(1/3*f[i,:]+f[i-1,:])*dx
        y=I;
    else:
       y=f*dx; 
    
    return y



	


#img_or=cv2.imread('rose_3.png', cv2.IMREAD_COLOR)
#gray_img_ref=cv2.cvtColor(img_or, cv2.COLOR_BGR2GRAY)

gray_img_ref=data.gravel()
img_dim_ref=400
img_o=gray_img_ref[0:img_dim_ref,0:img_dim_ref]

plt.imshow(img_o,cmap='gray')
plt.axis('off')
plt.show()
img=jnp.array(img_o,dtype='float32')


fe=1;

#gm=1.72
beta=1;
max_r=jnp.amax(img,1).reshape(-1,1)
max_c=jnp.amax(img,0).reshape(1,-1)

h_min_row=beta*(1/math.pi)*jnp.sqrt(max_r);
h_min_column=beta*(1/math.pi)*jnp.sqrt(max_c);

h_min=jnp.max((h_min_row+jnp.transpose(h_min_column))/2);

gm=jnp.log2(h_min)



prof = profile.Profile()
prof.enable()
img_scsa,psinnory,psinnorx,V1,kappax,kappay,Nx_v,Ny_v=SCSA2D2(img,h_min,gm,fe)
prof.disable()


stats = pstats.Stats(prof).strip_dirs().sort_stats("cumtime")
stats.print_stats(50) #



#%%

plt.imshow(img_scsa.astype('uint8'),cmap='gray')
plt.axis('off')
plt.show()

# mse=mean_squared_error(img,img_scsa) 
# psnr=peak_signal_noise_ratio(img,img_scsa,data_range=(255))
# ssim=structural_similarity(img,img_scsa,data=255) 