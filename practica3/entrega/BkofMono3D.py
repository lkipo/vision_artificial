import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fftshift, ifftshift

# Intentamos empregar a fft2 do modulo pyfftw se esta dispoñible
try:
    from pyfftw.interfaces.scipy_fftpack import fft2, ifft2, fftn, ifftn
# De outra forma, executarase scipy fftpack(~2-3x mais lenta!)
except ImportError:
    import warnings
    warnings.warn("""Modulo 'pyfftw' (FFTW Python bindings) sen intalar. Executa
                    'pip install pyfftw' no teu entorno""")
    from scipy.fftpack import fft2, ifft2
    

class BkofMono3D:
    def __init__(self, imname, nscale=2, minWaveLength=10, mult=2.1, sigmaOnf=0.55, black=False):
        self.img = []
        self.nscale = nscale
        # self.norient = norient
        self.minWaveLength = minWaveLength
        self.mult = mult
        self.sigmaOnf = sigmaOnf
        # self.dThetaOnSigma = dThetaOnSigma
        self.black = black
        self.even = []
        self.oddx = []
        self.oddy = []
        self.oddz = []

        self.img = []
        cap = cv.VideoCapture(imname)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Lectura de video completada...")
                break
            
            image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            
            #Cambiamos tipo da matriz da imaxe
            if image.dtype not in ['float32', 'float64']:
                image = np.float64(image)

            if image.ndim == 3:   #se e de cor promedio as bandas
                image = image.mean(2)

            if self.black:
                image = cv.bitwise_not(image)

            self.img.append(image)
            
    def filters3D(self):
        VIDF = fftn(self.img)
        self.img = VIDF
        rows, cols, frames = VIDF.shape
        
        if (cols % 2):
            xvals = np.arange(-(cols - 1) / 2.,
                            ((cols - 1) / 2.) + 1) / float(cols - 1)
        else:
            xvals = np.arange(-cols / 2., cols / 2.) / float(cols)

        if (rows % 2):
            yvals = np.arange(-(rows - 1) / 2.,
                            ((rows - 1) / 2.) + 1) / float(rows - 1)
        else:
            yvals = np.arange(-rows / 2., rows / 2.) / float(rows)

        if (rows % 2):
            zvals = np.arange(-(frames - 1) / 2.,
                            ((frames - 1) / 2.) + 1) / float(frames - 1)
        else:
            zvals = np.arange(-frames / 2., frames / 2.) / float(frames)

        x, y, z = np.meshgrid(xvals, yvals, zvals, sparse=True)
        
        # print(x)
        radius = np.sqrt(x*x + y*y + z*z)
        radius = ifftshift(radius)
        # theta = np.arccos(z/radius)
        # phi = np.arctan2(y, x)
        # print(theta.shape)
        # radius = ifftshift(radius)
        # theta = ifftshift(theta)
        # phi = ifftshift(phi)
        
        radius[0, 0] = 1.
        # print(x, y, z)
        # print(VIDF.shape)
        
        # sintheta = np.sin(theta) ##POSIBLE SOBRANTE
        # lp = self.__lowpassfilter((rows, cols, frames), .45, 15)

        logGaborDenom = 2. * np.log(self.sigmaOnf) ** 2.

        # for ss in range(self.nscale):
        #longura de onda para cada escala onde mult é 
        #a distancia entre filtros experado en octavas 1 octava = log2(w1/w2)
        wavelength = self.minWaveLength * self.mult ** (self.nscale)

        # Frecuencia central do filtro
        fo = 1. / wavelength
        
        # log Gabor
        logRadOverFo = np.log(radius / fo)
        even = np.exp(-(logRadOverFo * logRadOverFo) / logGaborDenom)
        # print('shaperadius', radius.shape)
        # print('shapelp', lp.shape)
        # print('shapeeven', even.shape)
        
        # Aplicamos o filtro paso-baixo para evitar o anillado
        # even = even * lp
        # Aseguramonos de que o punto de frecuenca o vale cero!
        even[0, 0] = 0.

        self.even = (even)
        self.oddx = (-1j*(x/radius)*even) # 1j*(x/radius)*even
        self.oddy = (-1j*(y/radius)*even)
        self.oddz = (-1j*(z/radius)*even)

    def getVideoResponse(self):

        # conv = self.even*self.img
        # space = ifftn(conv)
        # space = space.real
        # for frame in space:
        #     norm = np.linalg.norm(frame)
        #     cv.imshow('freim', (frame)*255/norm)
        #     # cv.imshow('frame', cv.normalize(frame, np.zeros_like(frame), 0, 255, cv.NORM_MINMAX))
        #     cv.waitKey(20)
        
        oddfilters = [self.even, self.oddx, self.oddy, self.oddz]
        for filter in oddfilters:
            conv = filter*self.img
            space = ifftn(conv)
            space = space.real

            for frame in space:
                norm = np.linalg.norm(frame)
                cv.imshow('resposta ao video', (frame)*255/norm)
                # cv.imshow('frame', cv.normalize(frame, np.zeros_like(frame), 0, 255, cv.NORM_MINMAX))
                cv.waitKey(20)
            
    def showFrecFilters(self):
        plt.imshow(self.even[len(self.even)//2])
        plt.show()
        for filter in self.even:
            cv.imshow('frame', cv.normalize((filter), np.zeros_like(filter), 0, 255, cv.NORM_MINMAX))
            cv.waitKey(20)
                
        for filter in self.oddx:
            cv.imshow('frame2', cv.normalize((filter.imag), np.zeros_like((filter.imag)), 0, 255, cv.NORM_MINMAX))
            cv.waitKey(20)

        for filter in self.oddy:
            cv.imshow('frame3', cv.normalize((filter.imag), np.zeros_like((filter.imag)), 0, 255, cv.NORM_MINMAX))
            cv.waitKey(20)


        for filter in self.oddz:
            cv.imshow('frame4', cv.normalize((filter.imag), np.zeros_like((filter.imag)), 0, 255, cv.NORM_MINMAX))
            cv.waitKey(20)
            
    def showSpaceFilters(self):
        for filter in self.even:
            cv.imshow('even', cv.normalize(ifftshift(ifftn(filter).real), np.zeros_like(ifftshift(ifftn(filter)).real), 0, 255, cv.NORM_MINMAX))
            cv.waitKey(20)
                
        for filter in self.oddx:
            cv.imshow('odd1', cv.normalize(ifftshift(ifftn(filter).imag), np.zeros_like(ifftshift(ifftn(filter).imag)), 0, 255, cv.NORM_MINMAX))
            cv.waitKey(20)

        for filter in self.oddy:
            cv.imshow('odd2', cv.normalize(ifftshift(ifftn(filter).imag), np.zeros_like(ifftshift(ifftn(filter).imag)), 0, 255, cv.NORM_MINMAX))
            cv.waitKey(20)


        for filter in self.oddz:
            cv.imshow('odd3', cv.normalize(ifftshift(ifftn(filter).imag), np.zeros_like(ifftshift(ifftn(filter).imag)), 0, 255, cv.NORM_MINMAX))
            cv.waitKey(20)
