# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:39:07 2024

@author: jack radford (jack.radford@glasgow.ac.uk)
"""
#pip install numba
import matplotlib.pyplot as plt
import numpy as np
from numba import njit

class tools:
    
    def __init__(self, params):
        # assign required global variables depending on the params object
        
        self.c=params['c']/params['n']
        self.mu_a=params['mu_a']
        self.mu_s=params['mu_s']
        self.bins=params['bins']
        self.pad_time=params['pad_time']
        self.res=params['res']
        self.FOV=params['FOV']
        self.t_res=params['t_res']
        self.st_bin=params['st_bin']
        self.nd_bin=params['nd_bin']
        
        self.a=np.zeros([params['pad_dim'],params['pad_dim'], params['pad_time']*2+params['bins'] ])    
        self.pad_start=int((params['pad_dim']/2)-(params['res']/2))
        self.pad_end=int((params['pad_dim']/2)+(params['res']/2))
        self.pad_FOV=params['pad_dim']*(params['FOV']/params['res'])
        self.pad_dim=params['pad_dim']
        
        
        if 'dOmega' in params.keys():
            self.dOmega = params['dOmega']
            self.I_I0 = params['I_I0'] # ratio of intensity out/in (I/I0)
        if 'delay' in params.keys():
            self.delay = params['delay']
        elif 'delay' not in params.keys():
            self.delay=0

    def PSF(self, slab_thickness, mu_a_only=0, semi_inf=False):
        # Calculate the PSF through the slab
        pad_FOV = self.pad_FOV
        pad_dim = self.pad_dim
 
        x, y=np.meshgrid(np.linspace(-1*(pad_FOV),pad_FOV,pad_dim)/2,np.flip(np.linspace(-1*(pad_FOV),pad_FOV,pad_dim)/2,0))
        
        if mu_a_only==1:
            psf = self.jitPSF_mu_a(slab_thickness,
                               x,
                               y,
                               self.mu_a,
                               self.mu_s,
                               self.t_res,
                               self.bins,
                               self.delay,
                               self.c,
                               self.a.shape,
                               self.dOmega,
                               self.I_I0
                               )
            
        elif semi_inf==True:
            psf1 = self.jitPSF(slab_thickness-(1/self.mu_s),
                               x,
                               y,
                               self.mu_a,
                               self.mu_s,
                               self.t_res,
                               self.bins,
                               self.delay,
                               self.c,
                               self.a.shape,
                               )
            psf2 = self.jitPSF(slab_thickness+(1/self.mu_s),
                               x,
                               y,
                               self.mu_a,
                               self.mu_s,
                               self.t_res,
                               self.bins,
                               self.delay,
                               self.c,
                               self.a.shape,
                               )
            psf = psf1-psf2
        else:
            psf = self.jitPSF(slab_thickness,
                               x,
                               y,
                               self.mu_a,
                               self.mu_s,
                               self.t_res,
                               self.bins,
                               self.delay,
                               self.c,
                               self.a.shape,
                               )
        return psf
    
    @staticmethod
    @njit
    def jitPSF(slab,
               x,
               y,
               mu_a,
               mu_s,
               t_res,
               bins,
               delay,
               c,
               a_shape):
         # Calculate the PSF through the slab


        D=1/(3*(mu_a+mu_s))  # define diffusion coefficient
        t=np.linspace(1e-12+delay,t_res*bins+delay,bins) # create a time vector
        
        # find the radial vector for the plane at the slab exit
        r=np.sqrt((x**2+y**2+slab**2))
        
        # find PSF using diffusion equation
        psf=np.zeros(a_shape)
        for i in range(len(t)): 
            psf[:,:,i]=(c/((4*np.pi*D*c*t[i])**(3/2)))*(np.exp((-(r**2))/(4*D*c*t[i])))*np.exp(-mu_a*c*t[i])
        
        return psf
    
    @staticmethod
    @njit
    def jitPSF_mu_a(slab,
               x,
               y,
               mu_a,
               mu_s,
               t_res,
               bins,
               delay,
               c,
               a_shape,
               dOmega,
               I_I0):
         # Calculate the PSF through the slab


        D=(mu_a*(slab**2))/(np.log(4*np.pi*I_I0/dOmega))**2  # define diffusion coefficient
        t=np.linspace(1e-12+delay,t_res*bins+delay,bins) # create a time vector
        
        # find the radial vector for the plane at the slab exit
        r=np.sqrt((x**2+y**2+slab**2))
        
        # find PSF using diffusion equation
        psf=np.zeros(a_shape)
        for i in range(len(t)): 
            psf[:,:,i]=(c/((4*np.pi*D*c*t[i])**(3/2)))*(np.exp((-(r**2))/(4*D*c*t[i])))*np.exp(-mu_a*c*t[i])
        return psf

    def pulse(self, mean, st_dev, start_bin, pix_illumination=0, flat_illumination=0, tophat_illumination=0, power=1, wavelength=808e-9, pulse_duration=120e-15, rep_rate=80e6):
        """
        Creates a laser pulse in time and space. 
        Assumes that the pulse duration is less than one timebin.
        Returns a 3D data cube pulse 
        """
        meanx = mean[0]
        meany = mean[1]
        return self.jitpulse(meanx,
                             meany,
                             st_dev,
                             start_bin,
                             pix_illumination,
                             flat_illumination,
                             tophat_illumination,
                             self.a.shape,
                             self.pad_FOV,
                             self.pad_dim,
                             self.pad_start,
                             self.pad_end
                             )
    
    @staticmethod
    @njit
    def jitpulse(meanx,
                 meany,
                 st_dev,
                 start_bin,
                 pix_illumination,
                 flat_illumination,
                 tophat_illumination,
                 a_shape,
                 pad_FOV,
                 pad_dim,
                 pad_start,
                 pad_end
                 ):
        
        pulse=np.zeros(a_shape)
        padx=np.linspace(-pad_FOV/2,pad_FOV/2,pad_dim)
        pady=np.linspace(-pad_FOV/2,pad_FOV/2,pad_dim)
        pad_centre = int(pad_dim/2)
        if flat_illumination==1:
            pulse[:,:,start_bin] = np.ones((pulse.shape[0], pulse.shape[1]))
            
        elif pix_illumination==1:
            pulse[int(meanx+pad_centre),int(meany+pad_centre),start_bin] = 1
        
        elif tophat_illumination==1:
            x = np.zeros((pad_dim, pad_dim))
            y = np.zeros((pad_dim, pad_dim))
            for i in range(pad_dim): #create a meshgrid with locations 
                x[i] = padx
                y[:,i] = padx
            dists = np.sqrt((x)**2+(y)**2) #calc distances from centre
            idx = np.where(dists<=st_dev) # find indices of distances less than the st_dev
            for i in range(len(idx[0])):
                pulse[idx[0][i], idx[1][i],  start_bin] = 1 # set the pulse equal to themask at starting bin
        
        else:
            # make Gaussian pulse
            for i in range(pad_dim):
                for j in range(pad_dim):
                    pulse[i,j,start_bin]=np.exp(-(padx[i]-meanx)**2/(st_dev)**2)*np.exp(-(pady[j]-meany)**2/(st_dev)**2)
        
        return pulse
    
    @staticmethod   
    @njit
    def mask(phi, msk):
        
        phi_msk=np.zeros(phi.shape)
        for i in range(phi_msk.shape[2]):
            phi_msk[:,:,i]=phi[:,:,i]*msk
        return phi_msk
    
    def noise(self, phi2, noise_floor, peak):
        # add poisson noise to histogram
        self.phi2=np.uint8(phi2/np.amax(phi2)*peak)  #define a global variable to be used in the graphs later
        
        noise_hist=np.uint8((phi2/np.amax(phi2))*peak+noise_floor)  #create a stepwise integer version of the generate histogram
        
        self.phi2_noise=np.zeros(self.a.shape)  #create a noise histogram
        for i in range(self.a.shape[2]):
            self.phi2_noise[:,:,i]=np.random.poisson(noise_hist[:,:,i])
            
        # subtract the noise_flor and ensure no negative values
        self.phi2_noise=self.phi2_noise-noise_floor
        self.phi2_noise[self.phi2_noise<0]=0
        
        # crop the data to represent the dimesnions of the camera 
        #self.phi2_noise=self.phi2_noise[self.pad_start:self.pad_end,self.pad_start:self.pad_end,:]
        
        return self.phi2_noise
    
    def low_pix_mask(self, data, thresh=0.45):
        # creates a mask to zero all pixels who have less than the threshold of time integrated counts
        data_integrated = np.sum(data, axis=2)
        data_integrated[data_integrated<thresh*np.amax(data_integrated)]=0
        data_integrated[data_integrated>0]=1
        data_masked = np.zeros(data.shape)
        for i in range(data.shape[2]):
            data_masked[:,:,i]=data[:,:,i]*data_integrated
        return data_masked
        

    def fig(self,bgrnd, phi2, phi2_noise, Y):
        
        plt.figure()
        
        plt.subplot(2, 4, 1)
        plt.imshow(self.msk,interpolation='none')
        plt.title("Hidden Object")
        plt.subplot(2, 4, 2)
        plt.imshow(np.sum(self.phi_msk,2),interpolation='none')
        plt.title("Middle Surface")
        plt.subplot(2, 4, 3)
        plt.imshow(np.sum(phi2,2),interpolation='none')
        plt.title("End Surface")
        plt.subplot(2, 4, 4)
        plt.imshow(np.sum(phi2_noise,2),interpolation='none')
        plt.title("Cropped End Surface with noise")
        plt.subplot(2, 4, 5)
        plt.plot(np.sum(np.sum(phi2,0),0))
#        plt.axis([0, 251, 0, 70])
        plt.title("Simulated Data")
        plt.subplot(2, 4, 6)
        plt.plot(np.sum(np.sum(bgrnd,0),0), label = 'simulated background')
        plt.plot(np.sum(np.sum(phi2_noise,0),0),  label = 'simulated object data')
        plt.legend()
#        plt.axis([0, 251, 0, 70])
        plt.title("Raw Data")
        
        plt.subplot(2, 4, 7)
        plt.plot(np.sum(np.sum(Y,0),0), label = 'simulated background \n subtracted data')
        plt.title("Background subtracted \n data")
        
        plt.subplot(2, 4, 8)
        plt.imshow(np.sum(Y,2))
        plt.title("Raw Data End Surface\n (Background subtracted)")

        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        plt.show()
        
    def derivative(self, x, psf, I):
        """
        Apply the adjoint A*(A(x)-Y)=A*(Residual)
        The Adjoint/Hermitian-Transpose/Conjugate-Transpose/Dual operator should do the conjugate transpose operation of the original operator
        The trick is to apply the conjugate of each operation in reverse order to the forward operator 
            -------------------------------------------
            if A(x) is our operator then find A*(x) by:
                
            step 1: t1_a
            Last thing we did to forward model was iFFT so the first thing the Adjoint operator does is FFT
            FFT[x]
            
            step 2: t1_b
            the second to last thing was multiplying by F[PSF] so we need to multiply by the conjugate (time-reversed) FFT
            FFT*[PSF] x FFT[x]
            
            step 3: t1_c
            The third to last thing we did (second thing) was fourier transform so here we need to iFFT
            iFFT[FFT*[PSF] x FFT[x]]
            
            step 4: t1
            The first thing we did in the forward operator was multiply by the fluence after first slab (I). Here we need to multiply by the conjugate fluence
            I* x (iFFT[FFT[PSF x FFT[x]]])
            
            |In our case:       |
            |    x = residual   |
            |    PSF = A_psf1   |
            |    I = A_phi1     |
            
            [All inputs are assumed normalised padded and with full timebins (no timebin cropping!)] 
            -----------------------------------------
        """ 
        #crop the psf to match the cropped histogram of the raw data (already done in main file for I!)
        
        #step 1
        t1_a = (np.fft.fftn(x))
        t1_a = t1_a/np.size(t1_a)
#        
        #step 2
        t1_b = np.conj(np.fft.fftn(psf))*t1_a
        
        # step 3 
        t1_c = np.fft.fftshift(np.fft.ifftn(t1_b), axes=(0,1))
        t1_c = t1_c*np.size(t1_c)
        
        # step 4
        t1_d = np.conj(I)*t1_c
        # get rid of padding
        t1_d = np.real(t1_d[self.pad_start:self.pad_end,self.pad_start:self.pad_end,:]) 

        # sum the third dimension to show which pixels have the most different histograms
        t1 = np.sum(t1_d, axis = 2)
        
        return t1, t1_d
    
    def tvd(self,X):
        """Find the gradient of the total variation as per the old MatLab version.
        """
        #find the Total Variation
        dx, dy = np.gradient(X)
        dM = np.sqrt(dx**2+dy**2)
        dM[dM<np.amax(dM)*0.05]= np.amax(dM)*0.05
        
        # prevent dividing by zero on initial guess of all ones
        if np.sum(dM)==0:
            dM=np.ones(dM.shape)
            
        # divide every element by derivative numerator
        dx = dx/dM
        dy = dy/dM

        # take derivative of these (since we're looking to minimise cost function and this is part of the cost function.)
        dxx, dxy = np.gradient(dx) 
        dyx, dyy = np.gradient(dy)
        
        #find the derivative of the tv norm
        tvd =-(dxx+dyy)
        
        return tvd
    
        
  