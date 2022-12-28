import main as by
import funcion as fn

if __name__=='__main__':
    #actualizamos o CSV
    fn.saveDesc('../planeDataset/randGen/', 10)
    print('sexo')
    path = 'algo.csv'
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

    print(f"Dataset {1}, Train ACC = {by.accuracy_metric(data.trainLabel, predicted_train)}")
    print(f"Dataset {1}, test ACC = {by.accuracy_metric(data.testLabel, predicted_test)} \n")
