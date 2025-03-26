from flask import Flask, request, render_template
import numpy as np
import pickle
from PIL import Image
import io
import keras

app = Flask(__name__)

# Cargar el modelo entrenado
model = keras.models.load_model("model.h5")

# Lista de clases de EMNIST Balanced
classes = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"
]

def preprocess_image(image):
    image = image.convert("L")  # Convertir a escala de grises
    image = image.resize((28, 28))  # Redimensionar al tamaño esperado
    image = np.array(image) / 255.0  # Normalizar
    image = image.reshape(1, 28, 28, 1)  # Ajustar dimensiones
    return image

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            image = Image.open(io.BytesIO(file.read()))
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)[0]
            top_indices = np.argsort(prediction)[::-1][:2]
            first_char = classes[top_indices[0]]
            second_char = classes[top_indices[1]]
            first_prob = prediction[top_indices[0]] * 100
            second_prob = prediction[top_indices[1]] * 100
            return render_template(
                "result.html",
                first_char=first_char,
                second_char=second_char,
                first_prob=first_prob,
                second_prob=second_prob
            )
    return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=True)