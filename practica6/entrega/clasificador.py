import main as by
import funcion as fn
from time import time

if __name__=='__main__':
    #actualizamos o CSV
    t1 = time()
    print('actualizando .csv...')
    fn.saveDesc('../planeDataset/randGen/', 14)
    print('.csv actualizado')
    
    t2 = time()
    path = 'fourier.csv'
    rawData = by.load_data(path)
    data = by.Data(rawData, bias=False)
    #Instantacionamos e adestramos
    model_q = by.ByQuadratic(data)
    model_q.fix()

    #Predecimos co modelos adestrado os datos de adestramento
    # e test. Ollo: non se gardan os datos derivados do adestramento
    #Ti debes gardalos en disco e lelos cando os precises.
    predicted_train = model_q.predict(data.trainData)
    predicted_test = model_q.predict(data.testData)
    
    t3 = time()
    print(f"Dataset {1}, Train ACC = {by.accuracy_metric(data.trainLabel, predicted_train)}")
    print(f"Dataset {1}, test ACC = {by.accuracy_metric(data.testLabel, predicted_test)} \n")
    print('Tempo de xeración dos descritores =', t2-t1)
    print('tempo de execución do modelo =', t3-t2)
    print('Tempo total =', t3-t1)
    
    
    ### achar matriz de confusión, accuracy, precission, recall, F1
    