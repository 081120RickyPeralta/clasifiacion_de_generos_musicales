from flask import Flask, render_template, request
from CDGM import modeloCNN, modeloSVM

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/procesar_svm', methods=['POST'])
def procesar_svm():
    from sklearn.svm import SVC
    # Obtener los datos del formulario de SVM
    kernel_svm = request.form['kernel_svm']
    C_svm = float(request.form['C_svm'])
    coef0_svm = float(request.form['coef0_svm'])
    generos_svm = request.form.getlist('genero_svm[]')
    num_entrenamiento_ini = int(request.form['num_entrenamiento_ini_svm'])
    num_entrenamiento_fin = int(request.form['num_entrenamiento_fin_svm'])
    num_prueba_ini = int(request.form['num_prueba_ini_svm'])
    num_prueba_fin = int(request.form['num_prueba_fin_svm'])
    seleccionBD = request.form['database']
    
    
    etiquetas_svm, caracteristicas_svm = modeloSVM.entrenamientoSVM('mfcc', num_entrenamiento_ini, num_entrenamiento_fin, generos_svm, seleccionBD)
    
    svm_classifier = SVC(kernel=kernel_svm, C=C_svm, coef0=coef0_svm, probability=True)
    svm_classifier.fit(caracteristicas_svm, etiquetas_svm)
    
     # Generar matriz de confusión de SVM
    matriz_confusion_svm, promedio_clasificacion_correcta = modeloSVM.generarMatrizDeConfusionSVM(num_genero_prueba_ini=num_prueba_ini, num_genero_prueba_fin=num_prueba_fin, generos=generos_svm, svm_classifier=svm_classifier, db_seleccionada=seleccionBD)

    return render_template('matriz_confusion.html', matriz_confusion=matriz_confusion_svm, promedio=promedio_clasificacion_correcta)

@app.route('/procesar_cnn', methods=['POST'])
def procesar_cnn():
    # Obtener los datos del formulario de CNN
    max_longitud = 1250
    lotes = int(request.form['lotes'])
    epocas = int(request.form['epocas'])
    generos_cnn = request.form.getlist('genero_cnn[]')
    num_entrenamiento_ini = int(request.form['num_entrenamiento_ini_cnn'])
    num_entrenamiento_fin = int(request.form['num_entrenamiento_fin_cnn'])
    num_prueba_ini = int(request.form['num_prueba_ini_cnn'])
    num_prueba_fin = int(request.form['num_prueba_fin_cnn'])
    
    seleccionBD = request.form['database']

    # Cargar datos
    caracteristicas_entrenamiento, etiqueta_entrenamiento = modeloCNN.cargar_datos(generos_cnn, num_entrenamiento_ini, num_entrenamiento_fin, seleccionBD)
    caracteristicas_prueba, etiqueta_prueba = modeloCNN.cargar_datos(generos_cnn, num_prueba_ini, num_prueba_fin, seleccionBD)
    
    # Construir modelo
    forma_entrada = (caracteristicas_entrenamiento.shape[1], caracteristicas_entrenamiento.shape[2], 1)  # (n_mfcc, time_steps, channels)
    num_clases = len(generos_cnn)
    modelo = modeloCNN.construir_modelo(forma_entrada, num_clases)

    # Compilar, entrenar y evaluar modelo
    modelo_entrenado = modeloCNN.compilar_entrenar_evaluar_modelo(modelo, caracteristicas_entrenamiento, etiqueta_entrenamiento, caracteristicas_prueba, etiqueta_prueba, lotes, epocas)

    # Guardar el modelo entrenado
    # modelo_entrenado.save("modelo_entrenado.h5")
    # print("Modelo entrenado guardado exitosamente.")

    # Generar matriz de confusión de CNN
    matriz_confusion_cnn, promedio_clasificacion_correcta = modeloCNN.generar_matriz_confusion(modelo_entrenado, caracteristicas_prueba, etiqueta_prueba, generos_cnn)

    return render_template('matriz_confusion.html', matriz_confusion=matriz_confusion_cnn, promedio=promedio_clasificacion_correcta)

if __name__ == "__main__":
    app.run(debug=True)
