Clasificación multiclase de imágenes oculares del segmento anterior

Repositorio asociado al desarrollo de un sistema de clasificación multiclase de imágenes oculares del segmento anterior mediante técnicas de visión por computadora y aprendizaje profundo.

El proyecto integra modelos de aprendizaje profundo para identificar cuatro clases de imágenes oculares:

Cataratas
Conjuntivitis
Sano
Pterigión

Este repositorio contiene scripts desarrollados en Python para el procesamiento, clasificación y evaluación de imágenes oculares. El flujo general incluye la carga de imágenes, aplicación de transformaciones, predicción mediante modelos entrenados y cálculo de métricas de desempeño.

El objetivo principal es evaluar el comportamiento de distintos enfoques de clasificación aplicados a imágenes del segmento anterior del ojo, considerando métricas como exactitud, precisión, sensibilidad, F1-score, matriz de confusión y curvas precisión–sensibilidad.

Archivos principales
Codigo.py: contiene el código principal utilizado para el procesamiento o clasificación de imágenes.
Evaluar_clasificador.py: script para evaluar el desempeño del clasificador mediante métricas de clasificación.
Metodología general

El flujo de trabajo seguido en el proyecto puede resumirse en las siguientes etapas:

Carga de imágenes oculares.
Preprocesamiento de las imágenes.
Aplicación del modelo de clasificación.
Obtención de predicciones por clase.
Evaluación del desempeño del modelo.
Generación de métricas y gráficas de resultados.
Métricas consideradas

Para evaluar el desempeño de los modelos se emplearon las siguientes métricas:

Exactitud global
Precisión por clase
Sensibilidad por clase
F1-score
Matriz de confusión
Curvas precisión–sensibilidad
Precisión promedio por clase

Estas métricas permiten analizar tanto el desempeño general del clasificador como su comportamiento específico para cada una de las clases evaluadas.

Requisitos

Para ejecutar los scripts se requiere tener instalado Python y las siguientes librerías:

pip install numpy pandas matplotlib scikit-learn opencv-python torch torchvision pillow

Dependiendo del modelo utilizado, también puede ser necesario instalar:

pip install timm xgboost
Ejecución

Para ejecutar el código principal:

python Codigo.py

Para evaluar el clasificador:

python Evaluar_clasificador.py

Es necesario verificar y modificar las rutas de archivos dentro de cada script, de acuerdo con la ubicación local de las imágenes, etiquetas reales, predicciones o modelos entrenados.

Estructura esperada de datos

Los scripts de evaluación trabajan con archivos CSV que contienen, al menos, las siguientes columnas:

image, real_label, predicted_label

Cuando se generan curvas precisión–sensibilidad, también se requieren probabilidades por clase, por ejemplo:

prob_cataracts, prob_conjunctivitis, prob_healthy, prob_pterygium

o sus equivalentes en español:

prob_cataratas, prob_conjuntivitis, prob_sano, prob_pterigion
Clases utilizadas
Clase en inglés	Clase en español
cataracts	Cataratas
conjunctivitis	Conjuntivitis
healthy	Sano
pterygium	Pterigión
Resultados

Los resultados obtenidos permiten comparar el desempeño de diferentes modelos de clasificación aplicados a imágenes oculares. Las matrices de confusión y las curvas precisión–sensibilidad se emplean para analizar los aciertos, errores y comportamiento probabilístico de cada modelo ante distintos umbrales de decisión.

Nota

Las rutas incluidas en los scripts corresponden al entorno local de desarrollo. Para reproducir los experimentos en otro equipo, es necesario actualizar dichas rutas y verificar que los archivos de entrada mantengan la estructura esperada.

Autor

Repositorio desarrollado como parte de un trabajo de investigación sobre clasificación multiclase de imágenes oculares del segmento anterior mediante visión por computadora y aprendizaje profundo.
