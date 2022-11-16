import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fftshift, ifftshift

# Intentamos empregar a fft2 do modulo pyfftw se esta dispoñible
try:
    from pyfftw.interfaces.scipy_fftpack import fft2, ifft2
# De outra forma, executarase scipy fftpack(~2-3x mais lenta!)
except ImportError:
    import warnings
    warnings.warn("""Modulo 'pyfftw' (FFTW Python bindings) sen intalar. Executa
                    'pip install pyfftw' no teu entorno""")
    from scipy.fftpack import fft2, ifft2
    

class BkofMono:
    def __init__(self, img, nscale=4, norient=6, minWaveLength=3, mult=2.1, sigmaOnf=0.55, dThetaOnSigma=2., black=False):
        self.img = img
        self.nscale = nscale
        self.norient = norient
        self.minWaveLength = minWaveLength
        self.mult = mult
        self.sigmaOnf = sigmaOnf
        self.dThetaOnSigma = dThetaOnSigma
        self.black = black
        self.__bank = []
        self.__total_filters = []
        
    def loggabor(self):

        #Cambiamos tipo da matriz da imaxe
        if self.img.dtype not in ['float32', 'float64']:
            self.img = np.float64(self.img)

        if self.img.ndim == 3:   #se e de cor promedio as bandas
            self.img = self.img.mean(2)

        if self.black:
            self.img = cv.bitwise_not(self.img)
        #Uns parametros iniciais que precisamos
        rows, cols = self.img.shape

        # Achamos a transformda de Fourier da imaxe de entrada
        IM = fft2(self.img) 

        # Convertimos as coordenadas cartesianas do espectro de Fourier
        # a coordenadas polares planas (radio, angulo)
        # Inicializamos as matrices X e Y con rangos normalizados entre +/- 0.5
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

        x, y = np.meshgrid(xvals, yvals, sparse=True)

        #coordendas planas: radio e angulo polar (sentido antihorario)
        radius = np.sqrt(x * x + y * y)
        theta = np.arctan2(-y, x)

        # Desprazamento de cuadrantes de  radius e theta para construir 
        # os filtros coa frecuencia 0 nas esquinas (formatos datos de fft2())
        radius = ifftshift(radius)
        theta = ifftshift(theta)

        # Como traballaremos con Gabor logaritmicas non pode
        # haber puntos nulos pola indeterminación da funcion logaritmo.
        # Polo tanto, que o orixe o cambiamos a 1 para que o log(1)=0.
        radius[0, 0] = 1.

        #Achamos os senos e os cosenos en cada punto do especto correpondente
        sintheta = np.sin(theta)
        costheta = np.cos(theta)

        del x, y, theta #Libremos algo de memoria
        
        # filtro paso-baixo e parametros aceptables: radius .45, 'sharpness' 15
        lp = self.__lowpassfilter((rows, cols), 0.45, 15)

        radialden = 2. * np.log(self.sigmaOnf) ** 2

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
            rows, cols = size

        #Comprobamos a paridade de filas e columnas para construir a grella
        # para determinar a parte radial de cada punto no espazo frecuencial
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

        x, y = np.meshgrid(xvals, yvals, sparse=True)
        radius = np.sqrt(x * x + y * y)

        return ifftshift(1. / (1. + (radius / cutoff) ** (2. * n)))
