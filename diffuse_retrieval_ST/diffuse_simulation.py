# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:47:04 2024

@author: jack radford (jack.radford@glasgow.ac.uk)
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import constants as const
import diffuse_toolkit as tk


class sim_fit():
    
    def __init__(self, params, pix_illumination=0, flat_illumination=0, tophat_illumination=0, mu_a_only=0, semi_inf=True):
        self.params = params
        self.semi_inf = semi_inf
        # remember int always rounds down (this only works for odd resolution)
        self.pad_centre = int((self.params['pad_dim'])/2)
        if np.mod(params['pad_dim'], 2) == 0: # if padding is even subtract 1 for python indexing
            self.pad_centre = self.pad_centre-1
            
        self.pad_start = int((self.pad_centre)-((self.params['res'])/2))
        self.pad_end = int((self.pad_centre)+((self.params['res'])/2))
        self.pad_time = params['pad_time']
        self.bins = params['bins']
        # create a class instance with desired variables 
        self.sim = tk.tools(self.params)
        
        # find point spread function for slab 1 according to diffusion equation
        self.psf1=self.sim.PSF(self.params['slab_1'], mu_a_only=mu_a_only, semi_inf=self.semi_inf)
        self.psf1 = self.psf1/self.psf1.sum()
        self.mu_a_only=mu_a_only
        # create a gaussian pulse
        self.pulse=self.sim.pulse(self.params['beam_pos'], 
                        self.params['st_dev'],
                        self.params['pulse_st'],
                        pix_illumination=pix_illumination,
                        flat_illumination=flat_illumination,
                        tophat_illumination=tophat_illumination,
                        )
        # convolve pulse and PSF to get the output after 1st slab
        # self.phi1=np.fft.fftshift(np.real((1/self.psf1.size)*np.fft.ifftn((np.fft.fftn(self.pulse)*np.fft.fftn(self.psf1)))), axes=(0,1))
        # ### use ifftshift
        self.phi1=np.fft.fftshift(
            np.real((1/self.psf1.size)*np.fft.ifftn(
                (np.fft.fftn(np.fft.ifftshift(self.pulse, axes=(0,1))))*np.fft.fftn(np.fft.ifftshift(self.psf1, axes=(0,1))))), axes=(0,1))
        self.phi1[self.phi1<0]=0 # corect artefacts from FFT which give negative counts
        
        if self.bins < self.params['nd_bin']:
            raise Exception("The end bin (params['nd_bin']) is later than total histogram bins (params['bins'])!")
            
    def import_alphabet(self):
        """
        load 88799 emnist-letters images and save as "alphabet_binary.npz" 
        """
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files('crawford/emnist', path='E:/OneDrive - University of Glasgow/2nd year/Diffuse_LeastSq', unzip=True)
        # from zipfile import ZipFile
        # zipobj = ZipFile('emnist-letters-train.csv.zip', 'r')
        # zipobj.extractall()
        
        import pandas as pd
        alpha = pd.read_csv('emnist-letters-train.csv', delimiter=',')
        alpha2 = alpha.to_numpy()
        alpha2 = alpha2[:,1:785] # column 0 is the labels 1:785 is the image
        alphabet = np.transpose(np.reshape(alpha2, (alpha2.shape[0], 28,28)))
        alphabet_binary = np.round(alphabet/np.max(alphabet))
        alphabet_binary = 1-alphabet_binary
        np.savez('alphabet_binary', alphabet_binary=alphabet_binary)
    
    def simulate(self, masks, max_counts=100, print_out=0, norm=False, pad_inverse=0, too_thin_rule=1, shift_mask = [0,0], ignore_warning=0):
        if print_out==1:
            print('simulating data...')

        if len(masks.shape)<3:
            masks = np.expand_dims(masks, 0)    # correct dimensions if given only one mask to simulate
        
        self.result = np.zeros((masks.shape[0],self.params['res'], self.params['res'], self.params['nd_bin']-self.params['st_bin'] ))
        
            
        for i in range(masks.shape[0]):
            mask = masks[i,:,:]
            MASK=np.ones([self.params['pad_dim'],self.params['pad_dim']])    
            
            # sometimes we want to set the padding to be a part of the mask rather than let light through
            # kwarg "pad_inverse" will block light rather than let it through
            if pad_inverse==1:
                MASK=np.zeros([self.params['pad_dim'],self.params['pad_dim']])
                
            MASK[int(self.pad_start+shift_mask[0]):int(self.pad_end+shift_mask[0]),
                 int(self.pad_start+shift_mask[1]):int(self.pad_end+shift_mask[1])] = mask
            phi_msk=self.sim.mask(self.phi1, MASK)
            if print_out==1:
                print('{}/{}'.format(i, masks.shape[0]))
                
            # output of the second slab
            if too_thin_rule==0 or self.params['slab_2'] >= (3*(1/self.params['mu_s'])):
                psf2=self.sim.PSF(self.params['slab_2'], mu_a_only=self.mu_a_only, semi_inf=self.semi_inf)  #find another point spread function for the second slab (only required if slabs are different)
                psf2 = psf2/psf2.sum()
                # convolve fluence (with mask) with the second point spread function
                # phi2=np.fft.fftshift(np.real((1/psf2.size)*np.fft.ifftn(np.fft.fftn(phi_msk)*np.fft.fftn(psf2))), axes=(0,1))
                # use ifftshift
                phi2=np.fft.fftshift(
                    np.real((1/psf2.size)*np.fft.ifftn(
                        (np.fft.fftn(np.fft.ifftshift(phi_msk, axes=(0,1))))*np.fft.fftn(np.fft.ifftshift(psf2, axes=(0,1))))), axes=(0,1))
                phi2[phi2<0]=0 # corect artefacts from FFT which give negative counts

                Y =  phi2[int(self.pad_start+shift_mask[0]):int(self.pad_end+shift_mask[0]),
                 int(self.pad_start+shift_mask[1]):int(self.pad_end+shift_mask[1]), :-2*self.params['pad_time']]  # crop padding
                Y = Y[:,:,self.params['st_bin']:self.params['nd_bin']]  # crop to range
                
                self.result[i,:,:,:] = Y
                if norm==True:
                    self.result[i,:,:,:] =  self.result[i,:,:,:]/self.result[i,:,:,:].sum()
            else:

                Y =  phi_msk[int(self.pad_start+shift_mask[0]):int(self.pad_end+shift_mask[0]),
                 int(self.pad_start+shift_mask[1]):int(self.pad_end+shift_mask[1]),:-2*self.params['pad_time']] # crop padding
                Y = Y[:,:,self.params['st_bin']:self.params['nd_bin']]  # crop to range
                self.result[i,:,:,:] = Y
                if norm==True:
                    self.result[i,:,:,:] =  self.result[i,:,:,:]/self.result[i,:,:,:].sum()
                if (i==0 and ignore_warning==0):
                    print('*** slab2 is too thin for diffusion approx. (ignoring slab2) ***')
            if print_out==1:   
                if np.mod(i,50)==0:
                    print('simulated:{}'.format(i+1))
        
        return self.result
 
# %% play around with class and plot to trouble shoot

if __name__ == "__main__":
    
    print('running troubleshoot in sim_fit...')
    
    #%% set simulation parameters
    params= dict(mu_a=0.09,         # absorption coefficient (cm^-1)
                  mu_s=16.5,         # reduced scattering coefficient (cm^-1)
                  n=1.4,            #refractive index of material
                  slab_1=2.5,        # thickness of slab 1 (cm)
                  slab_2=2.5,        # thickness of slab 2 (cm)
                  FOV=5,           # square FOV of the camera (cm) 
                  res=32,            # Resolution of the camera
                  c=const.c*100, # speed of light in cm/s
                  bins=256,          # Number of timebins
                  t_res=55e-12,      # time resolution of the camera
                  pad_dim=3*32 ,    # padding resolution outwith FOV
                  pad_time=100,
                  beam_pos=[0,0],    # center point of the incident beam (cm)
                  st_dev=1.27,        # beam width (cm)
                  pulse_st=0,        # starting bin for the Gaussian pulse 
                  st_bin=0,         # starting bin of histogram 
                  nd_bin=256,        # end bin of histgram
                  lamb_grad = 1 ,    # set to 1 or zero to include derivative of (A(X)-Y) (this is used to troubleshoot regularisers)
                  )
        
    #%% run simulation
    a = np.ones((params['res'],params['res']))
    SIM = sim_fit(params)
    result = SIM.simulate(a)
    
    #%% plot results
    plt.imshow(result[0].sum(2))
    plt.figure()
    plt.plot(result[0].sum((0,1)))
   
   