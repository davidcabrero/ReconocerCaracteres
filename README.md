# Reconocimiento de Caracteres

Este proyecto es una aplicación web basada en Flask que permite reconocer caracteres escritos de números a mano utilizando un modelo de deep learning con redes neuronales entrenado con el dataset MNIST.

## 📌 Características
- Predicción de caracteres (números) a partir de imágenes.
- Interfaz web para cargar imágenes y obtener resultados.
- Modelo preentrenado almacenado en `model.pkl`.
- Preprocesamiento automático de imágenes.

## 🚀 Instalación y Ejecución
1. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```
2. Ejecutar la aplicación:
   ```bash
   python app.py
   ```
3. Abrir en el navegador: `http://127.0.0.1:5000`

## 📂 Estructura del Proyecto
```
│── app.py              # Código principal de la aplicación Flask
│── crearModelo.ipynb   # Notebook con la creación y entrenamiento del modelo
│── model.pkl           # Modelo entrenado en formato pickle
│── templates/
│   ├── home.html       # Página de inicio
│   ├── result.html     # Página de resultados
│── requirements.txt    # Dependencias del proyecto
```
