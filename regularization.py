# -*- coding: utf-8 -*-
"""


@author: inesm

The following code was used and modified for the Chambolle-Pock algorithm:




#  Copyright 2015 Pierre Paleo <pierre.paleo@esrf.fr>
#  License: BSD 2-clause Simplified
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


"""



from __future__ import division
import numpy as np
from math import sqrt
from scipy.signal import convolve2d as conv2, correlate2d as corr2
import scipy.io
import scipy.misc as misc
import matplotlib.pyplot as plt
from image_operators import *



def get_L(P, PT, data, n_it=10):
    '''
    Calculates the norm of operator K = [grad, P],
    i.e the sqrt of the largest eigenvalue of K^T*K = -div(grad) + P^T*P :
        ||K|| = sqrt(lambda_max(K^T*K))

    P : forward projection
    PT : back projection
    data : acquired sinogram
    '''
    x = PT(data)
    for k in range(0, n_it):
        x = PT(P(x)) - div(gradient(x))
        s = sqrt(norm2sq(x))
        x /= s
    return sqrt(s)



def chambolle_pock(data, W, Lambda,  n_it=100):
    '''
    Chambolle-Pock algorithm for the minimization of the objective function
        ||P*x - d||_2^2 + Lambda*TV(x)
    P : projection operator, P:=W
    PT : backprojection operator, P:=W.T
    Lambda : weight of the TV penalization (the higher Lambda, the more sparse is the solution)
    n_it : number of iterations
    return_energy: if True, an array containing the values of the objective function will be returned
    '''
    
    #Compute the projection operators
    P = lambda x: W*x 
    PT = lambda x: W.T*x
    
    #L : norm of the operator [P, Lambda*grad]
    #computed in power_method function
    L = get_L(P, PT, data, n_it=100)

    sigma = 1.0/L
    tau = 1.0/L
    x = 0*PT(data)
    p = 0*gradient(x)
    q = 0*data
    x_tilde = 0*x
    theta = 1.0
    
    
    
    
    #History of the energy values
    en = np.zeros(n_it)
    for k in range(0, n_it):
        # Update dual variables
        p = proj_l2(p + sigma*gradient(x_tilde), Lambda)
        q = (q + sigma*P(x_tilde) - sigma*data)/(1.0 + sigma)
        # Update primal variables
        x_old = x
        x = x + tau*div(p) - tau*PT(q)
        x_tilde = x + theta*(x - x_old)
        # Calculate norms
        fidelity = 0.5*norm2sq(P(x)-data)
        tv = norm1(gradient(x))
        energy = 1.0*fidelity + Lambda*tv
        en[k] = energy
        if k >0 and energy>en[k-1]: #Stopping criterion
            break
    
    print("Final state -> energy %e " % energy)
    return x





#Thresholding at 5%
def thresholding5(R, cumhist, bins):
    '''Locate the values of R corresponding to the 5% highest values
    input:
    R: ratio matrix
    cumhist: cumulative histogram of the ratio matrix R
    bins: values of the bins for the cumulative histogram of R
    Returns a mask to locate the 5% highest values'''
    for i in range(cumhist.shape[0]-1, 1, -1):
        if cumhist[i]>=0.95 and cumhist[i-1]<=0.95:
            t = bins[i]
    target = R>t
    return target.astype(int)
    





def compute_R(v, omega=50):
    '''Compute the ratio Rx and Ry for each pixel in the resampled image v
    v : irregularly resampled image
    omega : width of the window used to compute the empirical variance
    output : Rx, Ry'''
    
    Rx = np.zeros(v.shape)
    Ry = np.zeros(v.shape)
    nx, ny = v.shape
    
    for k in range(v.shape[0]):
        for l in range(v.shape[1]):
            sigma_r_x = np.sum(np.real(v[max(0,k-omega):min(nx-1,k+omega), l])**2)/(2*omega)
            sigma_i_x = np.sum(np.imag(v[max(0,k-omega):min(nx-1,k+omega), l])**2)/(2*omega)
            Rx[k,l] = ( (np.real(v[k, l])**2/sigma_r_x) + (np.imag(v[k, l])**2/sigma_i_x) )**0.5
            
            sigma_r_y = np.sum(np.real(v[k, max(0,l-omega):min(ny-1,l+omega)])**2)/(2*omega)
            sigma_i_y = np.sum(np.imag(v[k, max(0,l-omega):min(ny-1,l+omega)])**2)/(2*omega)
            Ry[k,l] = ( (np.real(v[k, l])**2/sigma_r_y) + (np.imag(v[k, l])**2/sigma_i_y) )**0.5
            
    return Rx, Ry









def get_R_weights(R, target, w=1, h=5):
    '''Compute the weights based on the ratio matrix. 
    We want the target to be in the top 5% values for Rx and Ry
    and we use R=Rx+Ry to keep the amplitude importance
    We build a cross patern around these top values 
    where we want to keep the values of R
    input:
    R = Rx+Ry ratio matrix
    target= binary mask to locate the top 5% values for both Rx and Ry
    h= hight of the cross patern
    w= width of the cross patern
    Return:
    Weights based on the ratio R'''
    nx, ny = target.shape
    mask = target.astype(int)
    
    for k in range(nx):
        for l in range(ny):
            if target[k, l]:
                mask[max(0,k-w):min(k+w+1, nx), max(0,l-h):min(l+h+1, ny)] = 1
                mask[max(0,k-h):min(k+h+1, nx), max(0,l-w):min(l+w+1, ny)] = 1
    return mask*R
