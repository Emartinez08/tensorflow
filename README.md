# Reconocimiento Facial con TensorFlow Serving y Kubernetes

Este proyecto implementa un **sistema de reconocimiento facial de alta precisión** que aprovecha lo último en infraestructura moderna: entrenar y desplegar modelos directamente en **Kubernetes** usando contenedores Docker y **TensorFlow Serving**.  

Gracias a Kubernetes, no solo puedes entrenar tu modelo de manera reproducible y escalable, sino que también servirlo de forma confiable y accesible desde cualquier cliente Python. Esto convierte tu proyecto en un ejemplo práctico de **MLOps**, donde el ciclo completo de entrenamiento, despliegue y consumo del modelo se ejecuta en un entorno controlado y automatizable.  

Con este enfoque, puedes:

- Entrenar modelos en contenedores aislados que pueden escalar horizontalmente según tus necesidades.
- Servir modelos con **TensorFlow Serving**, garantizando respuestas rápidas y consistentes.
- Integrar fácilmente nuevos datos o clases sin modificar la infraestructura.
- Acceder a tu modelo desde cualquier cliente Python a través de una API REST estándar.

---

## 📁 Estructura del proyecto

```text
faces/
│
├─ adrian/                 # Imágenes de la persona 'adrian'
├─ jessi/                  # Imágenes de la persona 'jessi'
├─ adsoft/                 # Imágenes de la persona 'adsoft'
├─ simon/                  # Imágenes de la persona 'simon'
├─ carlos/                 # Imágenes de la persona 'carlos'
├─ Enrique/                # Imágenes de la persona 'Enrique'
│
├─ faces.py                # Script de entrenamiento
├─ client.py               # Cliente para enviar imágenes y recibir predicciones
├─ label_encoder.pkl       # LabelEncoder generado durante el entrenamiento
├─ reconocimiento-rostro/  # Carpeta donde se guarda el modelo exportado
│   └─ 1/                 # Versión del modelo
└─ README.md
