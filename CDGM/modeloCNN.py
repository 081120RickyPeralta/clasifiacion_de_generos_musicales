import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix

def generar_espectrograma(ruta_archivo):
    """
    Genera el espectrograma de un archivo de audio dado.
    """
    audio, _ = librosa.load(ruta_archivo, sr=22050)  # Cargar el audio con la misma tasa de muestreo que se utilizó para entrenar el modelo
    mfcc = librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=20)

    # Asegurarse de que todas las características tengan la misma longitud
    max_longitud = 1250
    if mfcc.shape[1] < max_longitud:
        relleno = max_longitud - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, relleno)), mode='constant')
    elif mfcc.shape[1] > max_longitud:
        mfcc = mfcc[:, :max_longitud]

    # Agregar dimensión del canal
    mfcc = np.expand_dims(mfcc, axis=-1)

    return mfcc

def cargar_datos(generos, num_entrenamiento_ini, num_entrenamiento_fin, db_seleccionada):
    """
    Carga los datos de audio y sus etiquetas.
    """
    caracteristicas = []
    etiqueta = []
    
    if (db_seleccionada=='ballroom'):
        for genero in generos:
            for i in range(num_entrenamiento_ini, num_entrenamiento_fin):
                archivo_audio = f'audios/ballroom/{genero}/{genero} ({i}).wav'
                espectrograma = generar_espectrograma(archivo_audio)
                caracteristicas.append(espectrograma)
                etiqueta.append(genero)
    elif (db_seleccionada=='gtzan'):
        for genero in generos:
            for i in range(num_entrenamiento_ini, num_entrenamiento_fin):
                archivo_audio = f'audios/gtzan/{genero}/{genero}.{"{:05d}".format(i)}.wav'
                espectrograma = generar_espectrograma(archivo_audio)
                caracteristicas.append(espectrograma)
                etiqueta.append(genero)

    caracteristicas = np.array(caracteristicas)
    etiqueta = np.array(etiqueta)

    # Codificar etiquetas
    codificador = LabelEncoder()
    etiqueta_codificada = codificador.fit_transform(etiqueta)

    return caracteristicas, etiqueta_codificada

def construir_modelo(forma_entrada, num_clases):
    """
    Construye el modelo de red neuronal convolucional.
    """
    modelo = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=forma_entrada),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='leaky_relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='leaky_relu'),
        Dropout(0.5),
        Dense(num_clases, activation='softmax')
    ])

    return modelo

def compilar_entrenar_evaluar_modelo(modelo, caracteristicas_entrenamiento, etiqueta_entrenamiento, caracteristicas_prueba, etiqueta_prueba, lote, epocas, optimizador='adam', perdida='sparse_categorical_crossentropy'):
    """
    Compila, entrena y evalúa el modelo.
    """
    # Compilar modelo
    modelo.compile(optimizer=optimizador, loss=perdida, metrics=['accuracy'])

    # Entrenar modelo
    modelo.fit(caracteristicas_entrenamiento, etiqueta_entrenamiento, batch_size=lote, epochs=epocas, validation_data=(caracteristicas_prueba, etiqueta_prueba))

    # Evaluar modelo
    perdida, precision = modelo.evaluate(caracteristicas_prueba, etiqueta_prueba)
    print("Precisión del modelo:", precision)

    return modelo

def generar_matriz_confusion(modelo, caracteristicas_prueba, etiqueta_prueba, generos):
    """
    Genera y muestra la matriz de confusión.
    """
    # Generar matriz de confusión
    etiqueta_predicha = np.argmax(modelo.predict(caracteristicas_prueba), axis=-1)
    matriz_confusion_resultado = confusion_matrix(etiqueta_prueba, etiqueta_predicha)

    # Calcular la matriz de confusión en porcentaje
    matriz_confusion_porcentaje = (matriz_confusion_resultado / matriz_confusion_resultado.sum(axis=1)[:, np.newaxis]) * 100

    # Crear DataFrame de la matriz de confusión
    df_matriz_confusion = pd.DataFrame(matriz_confusion_porcentaje, index=generos, columns=generos)
    print("Matriz de Confusión (en %):")
    print(df_matriz_confusion)

    # Calcular el porcentaje de clasificación correcta para cada género y promediarlos
    promedio_clasificacion_correcta = np.mean(np.diag(matriz_confusion_porcentaje))
    print("\nPromedio de clasificación correcta de todos los géneros:", promedio_clasificacion_correcta)
    
    # Crear un diccionario con la matriz de confusión y los nombres de los géneros
    matriz_confusion = {genero: fila_porcentaje.tolist() for genero, fila_porcentaje in zip(generos, matriz_confusion_porcentaje)}

    return matriz_confusion, promedio_clasificacion_correcta
