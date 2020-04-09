# -*- coding: utf-8 -*-
"""
@author: inesm
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from PIL import Image, ImageSequence
import rasterio
import mvalab
from numpy.fft import fft, ifft, fft2, ifft2, fftshift, ifftshift
import nfft
import plotly.express as px


def get_p0(u, x0, y0, K, axis):
    '''Return the position of the maximum amplitude in the signal'''
    id_max = 0
    max_val = 0
    
    nx, ny = u.shape

    if axis==0:
        #Horizontal maximum, variation over the columns (y axis)
        for i in range(-K, K):
            #Maximum amplitude 
            val = np.abs(u[x0, (ny+y0+i)%ny])
            if val > max_val:
                max_val = val
                id_max = i


    if axis==1:
        #Vertical maximum, variation over the lines (x axis)
        for i in range(-K, K):
            #Maximum amplitude 
            val = np.abs(u[(nx+x0+i)%nx, y0])
            if val > max_val:
                max_val = val
                id_max = i

    
    return id_max










def TV(u, x0, y0, K, p0, axis):
    '''Compute the TV of the image u centered on x0, y0
    axis: 0 for horizontal TV and 1 for vertical TV
    p0: position where the signal is maximum'''
    TV_im = 0
    TV_re = 0
    
    nx, ny = u.shape
    
    if axis==0:
        #Horizontal TV, variation over the columns (axis y)
        for i in range(-K, K):
            if (i != p0) and (i+1 != p0):
                TV_re += np.abs(np.real(u[x0, (ny+y0+i+1)%ny]) - np.real(u[x0, (ny+y0+i)%ny])) 
                TV_im += np.abs(np.imag(u[x0, (ny+y0+i+1)%ny]) - np.imag(u[x0, (ny+y0+i)%ny]))
                
    if axis==1:
        #Vertical TV, variation over x
        for i in range(-K, K):
            if (i != p0) and (i+1 != p0):
                TV_re += np.abs(np.real(u[(nx+x0+i+1)%nx, y0]) - np.real(u[(nx+x0+i)%nx, y0]))
                TV_im += np.abs(np.imag(u[(nx+x0+i+1)%nx, y0]) - np.imag(u[(nx+x0+i)%nx, y0]))
                
    return TV_re, TV_im








def ffttranslate(u, t, axis):
    ''' Subpixellic translation of image u, for a translation t in [-1/2, 1/2]
    using fourier interpolation in 2D. 
    The axis specifies if we want to perform a horizontal (axis=0) 
    or vertical (axis=1) translation.
    u : input image is supposed to be square'''

    if axis==1:
        u = u.T
    uhat = fft2(u);

    N = u.shape[1];
    U  = ifftshift(np.arange(- (N//2),N - ( N//2) ))/N
    ut = np.real(ifft2(uhat*np.exp(2j*np.pi * U*t )))
    
    if axis==1:
        return ut.T
    return ut






def translate(im, N_translation, axis):
    '''Compute a stack of translated image to test in order to find the true displacement'''
    translation_li = np.linspace(-0.5, 0.5, num=N_translation)
    im_translated = []
    
    
    for t in translation_li:
        u = ffttranslate(im, t, axis)
        im_translated.append(u)
        
    return im_translated, translation_li






def find_displacement(u, N_translation, K, tv_history=None, ampli_weights=False):
    '''Compute the displacement maps Tx and Ty
    N_translation: number of tested translations
    K : width of the window used to compute the TV
    tv_history: if True returns the evolution of the TV for a given pixel
    ampli_weights: if True, returns the weights based on the amplitude of the TV'''
    
    
    im_translated_x, translation_li_x = translate(u, N_translation, axis=0)#horizontal translation
    im_translated_y, translation_li_y = translate(u, N_translation, axis=1)#vertical translation
    
    if tv_history!=None:
        history_x = []
        history_y = []
    
    
    displacement = np.zeros((u.shape[0], u.shape[1], 2))
    delta_tv_x = np.zeros(u.shape)
    delta_tv_y = np.zeros(u.shape)
    
    
    for k in range(u.shape[0]):
        for l in range(u.shape[1]):
            #Horizontal translation tx
            tv_min = np.inf
            tv_max = -np.inf
            
            for i in range(len(translation_li_x)):
                #Find the index of maximum amplitude for real and imaginary parts
                p0 = get_p0(im_translated_x[i], k, l, K, axis=0)
                
                #Horizontal displacement
                tv_re, tv_imag = TV(im_translated_x[i], k, l, K, p0, axis=0)
                tv_x = tv_re + tv_imag
                
                if (tv_history!=None) and (tv_history[0]==k) and (tv_history[1]==l):
                    history_x.append(tv_x)
                    
                if tv_min>tv_x:
                    tv_min = tv_x
                    displacement[k, l, 0] = translation_li_x[i]
            
                if tv_max<tv_x:
                    tv_max = tv_x
            
            if ampli_weights:
                delta_tv_x[k, l] = tv_max - tv_min 
                    
            #Vertical translation ty
            tv_min = np.inf
            tv_max = -np.inf
            
            for i in range(len(translation_li_y)):
                #Find the index of maximum amplitude for real and imaginary parts
                p0 = get_p0(im_translated_x[i], k, l, K, axis=1)
                
                #Horizontal displacement
                tv_re, tv_imag = TV(im_translated_y[i], k, l, K, p0, axis=1)
                tv_y = tv_re + tv_imag
                
                if (tv_history!=None) and (tv_history[0]==k) and (tv_history[1]==l):
                    history_y.append(tv_y)
                    
                if tv_min>tv_y:
                    tv_min = tv_y
                    displacement[k, l, 1] = translation_li_y[i]
                    
                if tv_max<tv_y:
                    tv_max = tv_y
            
            if ampli_weights:
                delta_tv_y[k, l] = tv_max - tv_min 
            
        
    if tv_history!=None:
        return displacement, history_x, history_y
    
    if ampli_weights:
        return displacement, delta_tv_x, delta_tv_y

    return displacement   







