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
    def __init__(self, img, nscale=2, minWaveLength=10, mult=2.1, sigmaOnf=0.55, black=False):
        self.img = img
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
        
        self.imgresponse = []
        
    def monogenic(self):

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
        # print(theta)
        del x, y, theta #Libremos algo de memoria

        # Iniciamos a construcion dun banco de filtros log-Gabor 
        # a diferentes orientacions e escalas 

        # Os filtros son separables en duas compoñentes:
        # 1) A compoñente even, controla o ancho de bando do filtro e a frecuencia
        #   central a que responderá o filtro
        #    
        # 2) A compoñenete angular, que controla a orientación a que responde o filtro

        # O filtro logGabor se construe multiplicando as dúas compeñentes 
        # e o filtro queda conformado no espazo de Fourier

        # Parte even do filtro... 
        # Primeiro construimos un filtro paso-baixo tan grande como sexa 
        # posible en función das dimensións da imaxe de entrada (mostras espectrais)
        # e que caia a cero nos bordes espectrais. Este filtro será multiplicado
        # por todos os logGabor para evitar truncamentos bruscos e evitar asi o
        # fenomeno do anillado (ringing) como vimos na teoría

        # filtro paso-baixo e parametros aceptables: radius .45, 'sharpness' 15
        lp = self.__lowpassfilter((rows, cols), .45, 15)

        #Denominador da exponencia even do logGabor
        logGaborDenom = 2. * np.log(self.sigmaOnf) ** 2.

        #Sigma da parte angular en funcion do numro 
        # de otientacion e o parametro self.dThetaOnSigma. 
        #Este parametro regula a superposicion angular
        # thetaSigma = np.pi / self.norient / self.dThetaOnSigma
        # LogGaborAngularDenom = 2. * thetaSigma ** 2

        #Lazo para percorrer as escalas fixadas
        for ss in range(self.nscale):
            #longura de onda para cada escala onde mult é 
            #a distancia entre filtros experado en octavas 1 octava = log2(w1/w2)
            wavelength = self.minWaveLength * self.mult ** (ss)

            # Frecuencia central do filtro
            fo = 1. / wavelength
            
            # log Gabor
            logRadOverFo = np.log(radius / fo)
            even = np.exp(-(logRadOverFo * logRadOverFo) / logGaborDenom)

            # Aplicamos o filtro paso-baixo para evitar o anillado
            even = even * lp
            # Aseguramonos de que o punto de frecuenca o vale cero!
            even[0, 0] = 0.
            # print(costheta)
            scaleresponse = []
            
            self.even.append(even)
            scaleresponse.append(even*IM)
            
            
            filterx = 1j*costheta*even
            scaleresponse.append(filterx*IM)
            self.oddx.append(filterx)
            
            filtery = 1j*sintheta*even
            scaleresponse.append(filtery*IM)
            self.oddy.append(filtery)
            
            self.imgresponse.append(scaleresponse)
            
    def cobertura(self):
        
        total = np.zeros_like(self.even[0])
        for filt in self.even:
            total = filt + total
            
        plt.imshow(ifftshift(total))
        plt.show()
            
    def cobertura_corte(self):
        
        total = np.zeros_like(self.even[0])
        for filt in self.even:
            total = cv.inRange(filt, 0.66, 0.68).astype(np.float64) + total
            
        plt.imshow(ifftshift(total))
        plt.show()

    def showFrecFilters(self):
        for filter in self.even:
            plt.imshow(filter)
            plt.show()
    
        for filter in self.oddx:
            plt.imshow(np.imag(filter))
            plt.show()

        for filter in self.oddy:
            plt.imshow(np.imag(filter))
            plt.show()
            
    def showSpaceFilters(self):
        for filter in self.even:
            plt.imshow(ifftshift(np.real(ifft2(filter))))
            plt.show()

        for filter in self.oddx:
            plt.imshow(ifftshift(np.real(ifft2(filter))))
            plt.show()

        for filter in self.oddy:
            plt.imshow(ifftshift(np.real(ifft2(filter))))
            plt.show()

    def showFilterResponse(self):
        for scale in self.imgresponse:
            for img in scale:
                plt.imshow(ifft2(img).real)
                plt.show()

    def reconstruct(self):
        total = np.zeros_like(self.imgresponse[0][0])
        for scale in self.imgresponse:
            total = total + ifft2(scale[0])
                
        plt.imshow(total.real, 'gray')
        plt.show()
        
    def localEnergy(self):
        for scale in self.imgresponse:
            plt.imshow(ifft2(scale[0]).real**2 + ifft2(scale[1]).real**2 + ifft2(scale[2]).real**2)
            plt.show()
    
    def localEnergyTotal(self):
        total = np.zeros_like(self.even[0])
        for scale in self.imgresponse:
            total = np.maximum(total, (ifft2(scale[0]).real**2 + ifft2(scale[1]).real**2 + ifft2(scale[2]).real**2))
        plt.imshow(total)
        plt.show()
        
    def localAmplitude(self):
        for scale in self.imgresponse:
            plt.imshow(np.sqrt(ifft2(scale[0]).real**2 + ifft2(scale[1]).real**2 + ifft2(scale[2]).real**2))
            plt.show()
    
    def localAmplitudeTotal(self):
        total = np.zeros_like(self.even[0])
        for scale in self.imgresponse:
            total = np.maximum(total, np.sqrt(ifft2(scale[0]).real**2 + ifft2(scale[1]).real**2 + ifft2(scale[2]).real**2))
        plt.imshow(total)
        plt.show()
            
    def localPhase(self):
        for scale in self.imgresponse:
            plt.imshow(np.arctan2(np.sqrt(ifft2(scale[1]).real**2 + ifft2(scale[2]).real**2), ifft2(scale[0]).real))
            plt.show()
    
    def localPhaseTotal(self):
        total = np.zeros_like(self.even[0])
        for scale in self.imgresponse:
            total = np.maximum(total, np.arctan2(np.sqrt(ifft2(scale[1]).real**2 + ifft2(scale[2]).real**2), ifft2(scale[0]).real))
        plt.imshow(total)
        plt.show()
            
    def localOrient(self):
        for scale in self.imgresponse:
            plt.imshow(np.arctan2(ifft2(scale[1]).real, ifft2(scale[2]).real))
            plt.show()
  
    def localOrientTotal(self):
        total = np.zeros_like(self.even[0])
        for scale in self.imgresponse:
            total = np.maximum(total, (np.arctan2(ifft2(scale[1]).real, ifft2(scale[2]).real)))
        plt.imshow(total)
        plt.show()
  
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

        x, y = np.meshgrid(xvals, yvals, sparse=True)
        radius = np.sqrt(x * x + y * y)

        return ifftshift(1. / (1. + (radius / cutoff) ** (2. * n)))