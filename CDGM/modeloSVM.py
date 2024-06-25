import librosa
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

def generar_espectrograma(ruta_archivo, tipo):
    # Cargar el archivo de audio utilizando librosa
    audio, frecuencia_muestreo = librosa.load(ruta_archivo)
    n_fft = 2048
    if tipo == 'mfcc':
        # Calcular los coeficientes cepstrales de frecuencia Mel (MFCC)
        mfcc = librosa.feature.mfcc(y=audio, sr=frecuencia_muestreo, n_mfcc=20)
        return mfcc

def extraer_caracteristicas(espectrograma):
    # Calcular la media y la desviación estándar del espectrograma
    media = np.mean(espectrograma)
    desviacion_estandar = np.std(espectrograma)
    # Calcular la energía total de la señal
    # energia = np.sum(np.square(espectrograma))
    
    # Recortar el espectrograma a la dimensión deseada
    espectrograma_recortado = espectrograma[:, :1250]
    
    # Convertir el espectrograma recortado en un vector
    espectrograma_vector = espectrograma_recortado.reshape(-1)
    
    return espectrograma_vector

def entrenamientoSVM(filtro, num_genero_entrenamiento_ini, num_genero_entrenamiento_fin, generos, db_seleccionada):
    etiquetas_svm = []
    caracteristicas_svm = []
    
    if (db_seleccionada=='ballroom'):
        for genero in generos:
            for i in range(num_genero_entrenamiento_ini, num_genero_entrenamiento_fin):
                archivo_audio = f'audios/ballroom/{genero}/{genero} ({i}).wav'
                espectrograma = generar_espectrograma(archivo_audio, filtro)
                caracteristicas = extraer_caracteristicas(espectrograma)
                caracteristicas_svm.append(caracteristicas)
                etiquetas_svm.append(generos.index(genero))
    
    elif (db_seleccionada=='gtzan'):
        for genero in generos:
            for i in range(num_genero_entrenamiento_ini, num_genero_entrenamiento_fin):
                archivo_audio = f'audios/gtzan/{genero}/{genero}.{"{:05d}".format(i)}.wav'
                espectrograma = generar_espectrograma(archivo_audio, filtro)
                caracteristicas = extraer_caracteristicas(espectrograma)
                caracteristicas_svm.append(caracteristicas)
                etiquetas_svm.append(generos.index(genero))
    
    return etiquetas_svm, caracteristicas_svm

def generarMatrizDeConfusionSVM(num_genero_prueba_ini, num_genero_prueba_fin, generos, svm_classifier, db_seleccionada):   
    etiquetas_reales = []
    etiquetas_predichas = []
    
    if (db_seleccionada=='ballroom'):
        for genero in generos:
            for i in range(num_genero_prueba_ini, num_genero_prueba_fin):
                archivo_audio = f'audios/ballroom/{genero}/{genero} ({i}).wav'
                espectrograma = generar_espectrograma(archivo_audio, "mfcc")
                caracteristicas = extraer_caracteristicas(espectrograma)
                etiqueta_real = genero
                etiqueta_predicha = svm_classifier.predict([caracteristicas])[0]
                etiquetas_reales.append(etiqueta_real)
                etiquetas_predichas.append(generos[etiqueta_predicha])
    elif (db_seleccionada=='gtzan'):
        for genero in generos:
            for i in range(num_genero_prueba_ini, num_genero_prueba_fin):
                archivo_audio = f'audios/gtzan/{genero}/{genero}.{"{:05d}".format(i)}.wav'
                espectrograma = generar_espectrograma(archivo_audio, "mfcc")
                caracteristicas = extraer_caracteristicas(espectrograma)
                etiqueta_real = genero
                etiqueta_predicha = svm_classifier.predict([caracteristicas])[0]
                etiquetas_reales.append(etiqueta_real)
                etiquetas_predichas.append(generos[etiqueta_predicha])
                
    

    # Calcular la matriz de confusión
    confusion_matrix_result = confusion_matrix(etiquetas_reales, etiquetas_predichas, labels=generos)
    
    # Calcular el número total de muestras por clase
    total_por_clase = confusion_matrix_result.sum(axis=1)[:, np.newaxis]
    
    # Calcular la matriz de confusión en porcentaje
    confusion_matrix_porcentaje = (confusion_matrix_result / total_por_clase) * 100
    
    # Crear un DataFrame de Pandas con la matriz de confusión y los nombres de los géneros
    confusion_df = pd.DataFrame(confusion_matrix_porcentaje, index=generos, columns=generos)
    
    print("Matriz de Confusión (en %):")
    print(confusion_df)
    
    # Calcular el porcentaje de clasificación correcta para cada género y promediarlos
    porcentajes_clasificacion_correcta = []
    for i, genero in enumerate(generos):
        correcto = confusion_matrix_porcentaje[i, i]
        porcentajes_clasificacion_correcta.append(correcto)
    
    promedio_clasificacion_correcta = np.mean(porcentajes_clasificacion_correcta)
    print("\nPromedio de clasificación correcta de todos los géneros:", promedio_clasificacion_correcta)
    
    
    # Crear un diccionario con la matriz de confusión y los nombres de los géneros
    matriz_confusion = {genero: fila_porcentaje.tolist() for genero, fila_porcentaje in zip(generos, confusion_matrix_porcentaje)}
    return matriz_confusion, promedio_clasificacion_correcta

# if __name__ == "__main__":
#     num_genero_entrenamiento_ini = 0  # Número de archivos por género de música para entrenamiento
#     num_genero_entrenamiento_fin = 50  # Número de archivos por género de música para entrenamiento
#     num_genero_prueba_ini = 51  # Número de archivos por género de música para entrenamiento
#     num_genero_prueba_fin = 99  # Número de archivos por género de música para entrenamiento

#     # Definir etiquetas para cada tipo de música
#     generos = ['Rock', 'Jazz', 'Classical', 'Blues', 'Country']

#     etiquetas_svm, caracteristicas_svm = entrenamientoSVM('mfcc', num_genero_entrenamiento_ini, num_genero_entrenamiento_fin, generos)
    
#     svm_classifier = SVC(kernel='poly', C=3, coef0=6, probability=True)
#     svm_classifier.fit(caracteristicas_svm, etiquetas_svm)
    
#     generarMatrizDeConfusionSVM(num_genero_prueba_ini, num_genero_prueba_fin, generos, svm_classifier)
