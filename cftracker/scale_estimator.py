import numpy as np
import scipy
import cv2
from numpy.fft import fft, ifft
from scipy import signal
from lib.eco.fourier_tools import resize_dft
from .feature import extract_hog_feature
from lib.utils import cos_window
from lib.fft_tools import ifft2,fft2

class LPScaleEstimator:
    def __init__(self,target_sz,config):
        self.learning_rate_scale=config.learning_rate_scale
        self.scale_sz_window = config.scale_sz_window
        self.target_sz=target_sz

    def init(self,im,pos,base_target_sz,current_scale_factor):
        w,h=base_target_sz
        avg_dim = (w + h) / 2.5
        self.scale_sz = ((w + avg_dim) / current_scale_factor,
                         (h + avg_dim) / current_scale_factor)
        self.scale_sz0 = self.scale_sz
        self.cos_window_scale = cos_window((self.scale_sz_window[0], self.scale_sz_window[1]))
        self.mag = self.cos_window_scale.shape[0] / np.log(np.sqrt((self.cos_window_scale.shape[0] ** 2 +
                                                                    self.cos_window_scale.shape[1] ** 2) / 4))

        # scale lp
        patchL = cv2.getRectSubPix(im, (int(np.floor(current_scale_factor * self.scale_sz[0])),
                                                 int(np.floor(current_scale_factor * self.scale_sz[1]))), pos)
        patchL = cv2.resize(patchL, self.scale_sz_window)
        patchLp = cv2.logPolar(patchL.astype(np.float32), ((patchL.shape[1] - 1) / 2, (patchL.shape[0] - 1) / 2),
                               self.mag, flags=cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)

        self.model_patchLp = extract_hog_feature(patchLp, cell_size=4)

    def update(self,im,pos,base_target_sz,current_scale_factor):
        patchL = cv2.getRectSubPix(im, (int(np.floor(current_scale_factor * self.scale_sz[0])),
                                                   int(np.floor(current_scale_factor* self.scale_sz[1]))),pos)
        patchL = cv2.resize(patchL, self.scale_sz_window)
        # convert into logpolar
        patchLp = cv2.logPolar(patchL.astype(np.float32), ((patchL.shape[1] - 1) / 2, (patchL.shape[0] - 1) / 2),
                               self.mag, flags=cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
        patchLp = extract_hog_feature(patchLp, cell_size=4)
        tmp_sc, _, _ = self.estimate_scale(self.model_patchLp, patchLp, self.mag)
        tmp_sc = np.clip(tmp_sc, a_min=0.6, a_max=1.4)
        scale_factor=current_scale_factor*tmp_sc
        self.model_patchLp = (1 - self.learning_rate_scale) * self.model_patchLp + self.learning_rate_scale * patchLp
        return scale_factor

    def estimate_scale(self,model,obser,mag):
        def phase_correlation(src1,src2):
            s1f=fft2(src1)
            s2f=fft2(src2)
            num=s2f*np.conj(s1f)
            d=np.sqrt(num*np.conj(num))+2e-16
            Cf=np.sum(num/d,axis=2)
            C=np.real(ifft2(Cf))
            C=np.fft.fftshift(C,axes=(0,1))

            mscore=np.max(C)
            pty,ptx=np.unravel_index(np.argmax(C, axis=None), C.shape)
            slobe_y=slobe_x=1
            idy=np.arange(pty-slobe_y,pty+slobe_y+1).astype(np.int64)
            idx=np.arange(ptx-slobe_x,ptx+slobe_x+1).astype(np.int64)
            idy=np.clip(idy,a_min=0,a_max=C.shape[0]-1)
            idx=np.clip(idx,a_min=0,a_max=C.shape[1]-1)
            weight_patch=C[idy,:][:,idx]

            s=np.sum(weight_patch)+2e-16
            pty=np.sum(np.sum(weight_patch,axis=1)*idy)/s
            ptx=np.sum(np.sum(weight_patch,axis=0)*idx)/s
            pty=pty-(src1.shape[0])//2
            ptx=ptx-(src1.shape[1])//2
            return ptx,pty,mscore

        ptx,pty,mscore=phase_correlation(model,obser)
        rotate=pty*np.pi/(np.floor(obser.shape[1]/2))
        scale = np.exp(ptx/mag)
        return scale,rotate,mscore

