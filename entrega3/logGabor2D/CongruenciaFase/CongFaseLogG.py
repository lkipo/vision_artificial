# Xose R. Fdez-Vidal (e-correo: xose.vidal@usc.es)
# Codigo docente para o grao de Robótica da EPSE de Lugo.
# Dept. Física Aplicada, University of Santiago, 
# GALIZA,  2022 

#  Bibliografia:
#  [1]   D. J. Field, "Relations Between the Statistics of Natural Images and the
#  Response Properties of Cortical Cells", Journal of The Optical Society of
#  America A, Vol 4, No. 12, December 1987. pp 2379-2394
#  [2]  P. Kovesi. Image Features from Phase Congruency. Videre: Journal of
#  Computer Vision Research.pp. 1-26, vol 1, n3, 1999.
#  [3] Hand du Buf, "Task 1a:  A new Gabor filter set design". Tutorial
#  University of Algarbe, Faro. Portugal. 1998.


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


def lowpassfilter(size, cutoff, n):
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


def loggabor(img, nscale=4, norient=6, minWaveLength=3, mult=2.1, sigmaOnf=0.55, dThetaOnSigma=2.):
    """
    Funcion para achar un banco de filtros logGabor.

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



     Notas sobre a configuración do filtro para obter unha cobertura uniforme do espectro


    """

    #Cambiamos tipo da matriz da imaxe
    if img.dtype not in ['float32', 'float64']:
        img = np.float64(img)
        imgdtype = 'float64'
    else:
        imgdtype = img.dtype

    if img.ndim == 3:   #se e de cor promedio as bandas
        img = img.mean(2)

    #Uns parametros iniciais que precisamos
    rows, cols = img.shape

    # Inicializamos os acumuladores necesarios
    filterE = {} #Parte par do filtro para cada escala e orientacion
    filterO = {} #Parte impar do filtro para cada escala e orientacion
    EOscaleAll = {} #Resposta filtros para cada escala e orientacion
    ceros = np.zeros((rows, cols), dtype='float64')
    filtersum = ceros.copy()
    EnergyLocal = ceros.copy() # Enerxía local da imaxe

    pcSum = ceros.copy()

    # Achamos a transformda de Fourier da imaxe de entrada
    IM = fft2(img) 

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
    lp = lowpassfilter((rows, cols), .45, 15)

    #Denominador da exponencia radial do logGabor
    logGaborDenom = 2. * np.log(sigmaOnf) ** 2.
    logGabor = []

    #Sigma da parte angular en funcion do numro 
    # de otientacion e o parametro dThetaOnSigma. 
    #Este parametro regula a superposicion angular
    thetaSigma = np.pi / norient / dThetaOnSigma
    LogGaborAngularDenom = 2. * thetaSigma ** 2

    #Lazo para percorrer as escalas fixadas
    for ss in range(nscale):
        #longura de onda para cada escala onde mult é 
        #a distancia entre filtros experado en octavas 1 octava = log2(w1/w2)
        wavelength = minWaveLength * mult ** (ss)

        # Frecuencia central do filtro
        fo = 1. / wavelength

        # log Gabor
        logRadOverFo = np.log(radius / fo)
        tmp = np.exp(-(logRadOverFo * logRadOverFo) / logGaborDenom)
        #cv2.imshow("radial",fftshift(tmp))

        # Aplicamos o filtro paso-baixo para o anillado
        tmp = tmp * lp
        # Aseguramonos de que o punto de frecuenca o vale cero!
        tmp[0, 0] = 0.

        logGabor.append(tmp) #Gardamos a parte radial do filtro


    # Lazo principal

    # Para cada orientacion...
    for oo in range(norient):

        # Construimos a parte angular do filtro logGabor
        angl = oo * (np.pi / norient)

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
        spread = np.exp( -(dtheta ** 2) / LogGaborAngularDenom)


        # Congruencia da fase ======================
        #Acumuladores para achas a congruencua da fase
        sumE_ThisOrient = ceros.copy()
        sumO_ThisOrient = ceros.copy()
        sumAn_ThisOrient = ceros.copy()
        Energy = ceros.copy()
        EOscale = []
        #==============================================

        # Para cada escala ...
        for ss in range(nscale):

            # Multiplicamos as compoñentes radial e angular para ter o filtro logGabor
            filt = logGabor[ss] * spread
            #cv2.imshow("LogGabor",fftshift(filt))
            #cv2.waitKey(0)

            # Convolucionamos a imaxe cos filtros par e impar
            # e voltamos ao dominio espacial
            thisEO = ifft2(IM * filt)
            
            # Almacenamos as respostas de cada filtro 
            # en otientacion e escala
            EOscaleAll[oo, ss]=thisEO

            # Enerxía local total da sinal (modulo)
            An = np.abs(thisEO)
            EnergyLocal += An

            #Congruencia da fase ===================
            #Suma de amplitudes nesta orientacion
            sumAn_ThisOrient += An 

            # Sumas das respostas dos filtros 
            # pares nesta orientacion
            sumE_ThisOrient += np.real(thisEO)

            # Sumas das respostas dos filtros 
            # impares nesta orientacion
            sumO_ThisOrient += np.imag(thisEO)

            # Resposta dos filtros nesta direccion
            EOscale.append(thisEO)

            # Achamos a maxima enerxia a traves das escalas
            if ss == 0:
                maxAn = An
            else: 
                maxAn = np.maximum(maxAn, An)
            #============================================


        # Congruencia da fase =======================
        # Sumamos os modulos (sen fase), para cada orientacion
        XEnergy = np.sqrt(sumE_ThisOrient * sumE_ThisOrient +
                          sumO_ThisOrient * sumO_ThisOrient)
        MeanE = sumE_ThisOrient / XEnergy
        MeanO = sumO_ThisOrient / XEnergy

        # Agora achamos An(cos(phase_deviation)-| sin(phase_deviation))| 
        # empregando o produto escalar e vectorial entre a media en cada
        # orientacion e a resposta de cada filtro indivisual.
        #A esta cantida é a congruencia da fase multiplicada por An
        # que chamamos Energy

        for ss in range(nscale):
            E = np.real(EOscale[ss])
            O = np.imag(EOscale[ss])
            Energy += E * MeanE + O * MeanO - np.abs(E * MeanO - O * MeanE)

        #Estratexia para eliminar o ruido
        T = 3
        g=10
        cutoff = 0.5
        Energy = np.maximum(Energy - T, 0)
        width = (sumAn_ThisOrient / (maxAn + 0.00001) - 1.) / (nscale - 1)
        weight = 1. / (1. + np.exp(g * ( cutoff- width)))
        thisPC =  weight * Energy / sumAn_ThisOrient
        pcSum += thisPC

    return pcSum

def main(args):

    #Lemos a imaxe en formato gris e visualizamos
    img = cv2.imread(args['image'],0)
    if img is None:
        print('Imaxe non atopada: {}'.format(args['image']))
        exit(0)
 


    #Construimos os filtros e facemos a convolucion con img
    PC = loggabor(img, nscale=3, norient=4, minWaveLength=8, 
             mult=2.1, sigmaOnf=0.55, dThetaOnSigma=2.2)

    #visualizamos o resultado
    cv2.imshow("Congruencia da fase",np.float32(cv2.normalize(np.float32(PC), None, 0, 1, cv2.NORM_MINMAX)))
    cv2.waitKey(0)


if __name__ == '__main__':
    # analizamos os argumentos de entrada
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Imaxe de entrada")
    args = vars(ap.parse_args())
    main(args)