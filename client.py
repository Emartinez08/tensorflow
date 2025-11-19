import json
import numpy as np
import cv2
import requests

# Endpoint de tu modelo
#SERVER_URL = "http://localhost:8501/v1/models/reconocimiento-rostro:predict"
SERVER_URL = "https://tensorflow-serving-faces.onrender.com/v1/models/reconocimiento-rostro:predict"
#IMAGE_PATH = "jessi/IMG_4673.JPG"  # Reemplaza con la ruta a tu imagen
#IMAGE_PATH = "Enrique/IMG_6100.JPG"
#IMAGE_PATH = "carlos/160.jpeg"
#IMAGE_PATH = "simon/IMG_3821.JPG"
#IMAGE_PATH = "adrian/IMG_4125.JPG"
IMAGE_PATH = "adsoft/IMG_4434.JPG"

def main():
    # Cargar imagen en escala de grises
    img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (150, 150))

    # Normalizar y adaptar dimensiones
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)  # (150,150,1)
    img = np.expand_dims(img, axis=0)   # (1,150,150,1)

    # JSON para TensorFlow Serving
    predict_request = json.dumps({"instances": img.tolist()})
    headers = {"content-type": "application/json"}

    # POST al servidor
    response = requests.post(SERVER_URL, data=predict_request, headers=headers)
    response.raise_for_status()

    # Procesar predicción
    prediction = response.json()["predictions"][0]
    clases = ['Enrique', 'adrian', 'adsoft', 'carlos', 'jessi', 'simon']
    # clases = ["adrian", "adsoft", "jessi", "simon"]
    print("Predicciones (probabilidades):", prediction)
    print("Predicción final:", clases[np.argmax(prediction)])

if __name__ == "__main__":
    main()
