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
        cap = cv.VideoCapture('datatest/videos/SCE_circulos_girando_3.avi')
        
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
        lp = self.__lowpassfilter((rows, cols, frames), .45, 15)

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
        print(radius)
        self.even = (even)
        self.oddx = (-1j*(x/radius)*even) # 1j*(x/radius)*even
        self.oddy = (-1j*(y/radius)*even)
        self.oddz = (-1j*(z/radius)*even)
        
    def getfilters(self): #NON USAR
        print(self.even)
        for frame in self.even:
            # for frame in filter:
            cv.imshow('Even 1', (ifftn(frame).real))
            cv.waitKey(30)

        for filter in self.oddx:
            for frame in filter:
                cv.imshow('odd 1', (ifftn(frame).real))
                cv.waitKey(30)

        for filter in self.oddz:
            for frame in filter:
                cv.imshow('odd 2', (ifftn(frame).real))
                cv.waitKey(30)

        for filter in self.oddz:
            for frame in filter:
                cv.imshow('odd 3', (ifftn(frame).real))
                cv.waitKey(30)

    def getvideoresponse(self):

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
            print(space.real)
            print(filter)
            for frame in space:
                norm = np.linalg.norm(frame)
                cv.imshow('freim', (frame)*255/norm)
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
            cv.imshow('frame', cv.normalize(ifftshift(ifftn(filter).real), np.zeros_like(ifftshift(ifftn(filter)).real), 0, 255, cv.NORM_MINMAX))
            cv.waitKey(20)
                
        for filter in self.oddx:
            cv.imshow('frame2', cv.normalize(ifftshift(ifftn(filter).imag), np.zeros_like(ifftshift(ifftn(filter).imag)), 0, 255, cv.NORM_MINMAX))
            cv.waitKey(20)

        for filter in self.oddy:
            cv.imshow('frame3', cv.normalize(ifftshift(ifftn(filter).imag), np.zeros_like(ifftshift(ifftn(filter).imag)), 0, 255, cv.NORM_MINMAX))
            cv.waitKey(20)


        for filter in self.oddz:
            cv.imshow('frame4', cv.normalize(ifftshift(ifftn(filter).imag), np.zeros_like(ifftshift(ifftn(filter).imag)), 0, 255, cv.NORM_MINMAX))
            cv.waitKey(20)

    def __lowpassfilter(self, size, cutoff, n):
        """
        Construimos un filtro paso-baixa de Butterworth con función de transferencia:

            f = 1 / (1 + (w/cutoff)^2n)

        usa:  f = lowpassfilter(sze, cutoff, n)

        onde:  size é unha tupla especificando o tamanho do filtro como [rows cols].
                cutoff  é a frecuencia de corte do filtro entre 0 - 0.5
                n   é o orde do filtro, canto mais alto é n mais abruta é a transicion
                (n debe ser enteiro par maior que >= 1).

        OLLO: o orixe do filtro devolto esta na esquina superior.
        """

        if cutoff < 0. or cutoff > 0.5:
            raise Exception('cutoff debe estar 0 e 0.5')
        elif n % 1:
            raise Exception('n debe par e >= 1')
        if len(size) == 1:
            rows = cols = size
        else:
            rows, cols, frames = size

        #Comprobamos a paridade de filas e columnas para construir a grella
        # para determinar a parte even de cada punto no espazo frecuencial
        #Normalizado entre [0,1]
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
            
        if (frames % 2):
            zvals = np.arange(-(frames - 1) / 2.,
                            ((frames - 1) / 2.) + 1) / float(frames - 1)
        else:
            zvals = np.arange(-frames / 2., frames / 2.) / float(frames)

        x, y, z = np.meshgrid(xvals, yvals, zvals, sparse=True)
        radius = np.sqrt(x*x + y*y + z*z)

        return ifftshift(1. / (1. + (radius / cutoff) ** (2. * n)))

def main():

    #Lemos a imaxe en formato gris e visualizamos
    img = cv.imread('datatest/apolo.png',0)

    Bank = BkofMono3D(img)
    Bank.filters3D()
    Bank.getvideoresponse()
    # Bank.getfilters()
    # Bank.loggabor()
    # Bank.showSpaceFilters()
    # Bank.showFrecFilters()
    
if __name__ == '__main__':
    main()