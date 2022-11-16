#Importamos as librerias precisas
import argparse
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
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

class BkofLoGabor:
    def __init__(self, img, nscale=4, norient=6, minWaveLength=3, mult=2.1, sigmaOnf=0.55, dThetaOnSigma=2., black=False):
        """
        Clase para achar un banco de filtros logGabor.

        Argumentos:
        -----------
        <Name>      <Default>   <Description>
        img             N/A     Imaxe de entrada
        nscale          5       numero de escalas, o normal un numero entre 3-6
        norient         6       numero de orientacions.
        minWaveLength   3       Longura de onda do filtro de maior frecuencia.
        mult            2.1     Factor para achar a separaion entre filtros de escalas sucesivas.
        sigmaOnf        0.55    Factor para determinas a desviación tipica
                                na parte radial do filtro.
                                sigmaOnf    .85   mult 1.3
                                sigmaOnf    .75   mult 1.6  (ancho de banda ~1 octava)
                                sigmaOnf    .65   mult 2.1
                                sigmaOnf    .55   mult 3    (ancho de banda ~2 octaves)

        dThetaOnSigma   2       Factor que permite regular a sigma do filtro
                                na parte angular. 
                                dThetaOnSigma >  2 menor superposicion filtros angularmente
                                dThetaOnSigma <  2 maior superposicion filtros angularmente
        """
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
            img = np.float64(self.img)
            imgdtype = 'float64'
        else:
            imgdtype = img.dtype

        if self.img.ndim == 3:   #se e de cor promedio as bandas
            self.img = self.img.mean(2)

        if self.black:
            self.img = cv2.bitwise_not(self.img)
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

        # Iniciamos a construcion dun banco de filtros log-Gabor 
        # a diferentes orientacions e escalas 

        # Os filtros son separables en duas compoñentes:
        # 1) A compoñente radial, controla o ancho de bando do filtro e a frecuencia
        #   central a que responderá o filtro
        #    
        # 2) A compoñenete angular, que controla a orientación a que responde o filtro

        # O filtro logGabor se construe multiplicando as dúas compeñentes 
        # e o filtro queda conformado no espazo de Fourier

        # Parte radial do filtro... 
        # Primeiro construimos un filtro paso-baixo tan grande como sexa 
        # posible en función das dimensións da imaxe de entrada (mostras espectrais)
        # e que caia a cero nos bordes espectrais. Este filtro será multiplicado
        # por todos os logGabor para evitar truncamentos bruscos e evitar asi o
        # fenomeno do anillado (ringing) como vimos na teoría

        # filtro paso-baixo e parametros aceptables: radius .45, 'sharpness' 15
        lp = self.__lowpassfilter((rows, cols), .45, 15)

        #Denominador da exponencia radial do logGabor
        logGaborDenom = 2. * np.log(self.sigmaOnf) ** 2.
        logGabor = []

        #Sigma da parte angular en funcion do numro 
        # de otientacion e o parametro self.dThetaOnSigma. 
        #Este parametro regula a superposicion angular
        thetaSigma = np.pi / self.norient / self.dThetaOnSigma
        LogGaborAngularDenom = 2. * thetaSigma ** 2

        #Lazo para percorrer as escalas fixadas
        for ss in range(self.nscale):
            #longura de onda para cada escala onde mult é 
            #a distancia entre filtros experado en octavas 1 octava = log2(w1/w2)
            wavelength = self.minWaveLength * self.mult ** (ss)

            # Frecuencia central do filtro
            fo = 1. / wavelength

            # log Gabor
            logRadOverFo = np.log(radius / fo)
            tmp = np.exp(-(logRadOverFo * logRadOverFo) / logGaborDenom)

            # Aplicamos o filtro paso-baixo para evitar o anillado
            tmp = tmp * lp
            # Aseguramonos de que o punto de frecuenca o vale cero!
            tmp[0, 0] = 0.

            logGabor.append(tmp) #Gardamos a parte radial do filtro


        # Lazo principal: monta a parte radial e angular de cada
        #filtro en funcion da sua escala e orientacion. Tamen
        # obten a convolucion de cada filtro coa imaxe dando como
        # resultado unha responsta comlexa no dominio espacial: na
        # parte real a resposta do filtro par e na imaxinaria a resposta
        # do filtro impar 

        # Para cada orientacion...
        bank = []
        
        for oo in range(self.norient):

            # Construimos a parte angular do filtro logGabor
            angl = oo * (np.pi / self.norient)

            # Calculamos a distancia angular para cada punto da matriz do filtro
            # desde a orientación do filtro especificada. Para superar discontinuidade
            # na distancia angular (360-->0) empregamos o truco trigonometrico 
            # que consiste en tomas as diferencais angulares en seno e en coseno e logo achar
            # a distancia angulos mediante o arcotanxente arctg2(seno/cos). Deste xeito,
            #o seno e o coseno absorben as discontinuidades angulas!
    
            # diferencias en seno e coseno
            ds = sintheta * np.cos(angl) - costheta * np.sin(angl)
            dc = costheta * np.cos(angl) + sintheta * np.sin(angl)

            # diferencia angular absoluta
            dtheta = np.abs(np.arctan2(ds, dc))
            # parte angular do filtro
            spread = np.exp( -(dtheta ** 2) / LogGaborAngularDenom)

            # Para cada escala ...
            
            # scale_mult = np.ones([rows, cols])
            scale = []
            for ss in range(self.nscale):

                # Multiplicamos as compoñentes radial e angular 
                # para construir o filtro logGabor 2D
                filt = logGabor[ss] * spread
                
                self.__total_filters.append(filt)
                # Convolucionamos a imaxe cos filtros par e impar
                # e voltamos ao dominio espacial. Na parte real de
                # thisEO temos a resposta do filtro par da logGabor
                # e na parte imaxinaria a parte impar da logGabor.
                thisEO = ifft2(IM * filt)

                # scale_mult = np.abs(np.multiply(scale_mult, np.real(thisEO)))
                # plt.imshow(np.real(thisEO))
                # plt.show()
                scale.append(np.real(thisEO))
            
            bank.append(scale)
        
        self.__bank = np.array(bank)
        # return bank # matriz de bancos. PREGUNTAR A XOSE

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

    def roseta(self):
        
        total = np.zeros_like(self.__total_filters[0])
        for filter in self.__total_filters:
            
            total = cv2.add(total, ifftshift(filter))
            
        
        plt.imshow(total)
        plt.show()
        
    def roseta_corte(self, margin):
        total = np.zeros_like(self.__total_filters[0], dtype=np.float64)
        for filter in self.__total_filters:
            maximo = filter.max()
            range = cv2.inRange(filter, 0.67-margin, 0.67+margin).astype(np.float64)
            range = range

            total = cv2.add(total, ifftshift(range))
        plt.imshow(total)
        plt.show()
        
    def spacial_filters(self):
        for filter in self.__total_filters:
            plt.imshow(ifftshift(np.real(ifft2(filter))))
            plt.show()
                
    def radial_symmetry(self):
        mult_scale = np.zeros_like(self.__bank[0][0])
        for i in range(self.nscale):
        
            scale = self.__bank[:, i]
            mult_orientation = np.ones_like(self.__bank[0][0])
            
            for orient in scale:
                mult_orientation = mult_orientation * np.abs(orient)
            
            # plt.imshow(mult_orientation)
            # plt.show()
            
            mult_scale = np.maximum(mult_scale, mult_orientation)
        
        plt.imshow(mult_scale)
        plt.show()
        
        cv2.imwrite('simetria.jpg', cv2.normalize(mult_scale, np.zeros_like(mult_scale), 0, 255, cv2.NORM_MINMAX))
            
    def reconstruct(self):
        total = np.zeros_like(self.__bank[0][0])
        for i in self.__bank:
            for filter in i:
                total = total + filter
                
        plt.imshow(total, 'gray')
        plt.show()

    def local_energy(self):
        total = np.zeros_like(self.__bank[0])
        for filter in self.__bank:
            print(filter)
    
def main(args):

    #Lemos a imaxe en formato gris e visualizamos
    img = cv2.imread(args['image'],0)
    if img is None:
        print('Imaxe non atopada: {}'.format(args['image']))
        exit(0)


    # Construimos os filtros cos parametros que definen
    # o banco de filtros logGabor (cobertura espectral) e
    # pasamoslle a imaxe para convolucionar os filtros coa
    # imaxe e obter as respostas pares e impates
    Bank = BkofLoGabor(img, nscale=3, norient=6, minWaveLength=8, 
             mult=2.1, sigmaOnf=0.6, dThetaOnSigma=2)
    
    Bank.loggabor()
    # Bank.roseta()
    # Bank.roseta_corte(0.02)
    # Bank.radial_symmetry()
    # Bank.spacial_filters()
    # Bank.reconstruct()
    # Bank.local_energy()
            
if __name__ == '__main__':
    # analizamos os argumentos de entrada
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=False, help="Imaxe de entrada", default='datatest/apolo.png')
    ap.add_argument("-b", "--black", required=False, help="Fiduciarios de cor negra", default='False', type=bool)
    # ENGADIR ARGUMENTO PARA BLACK=FALSE/TRUE
    args = vars(ap.parse_args())
    main(args)