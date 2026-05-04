# 🔗 Detección de SPAM en URLs - Aplicación Web con Streamlit

Una aplicación web de machine learning que detecta si una URL es spam o no, utilizando Procesamiento de Lenguaje Natural (NLP) y un clasificador de Máquinas de Vectores de Soporte (SVM).

**Aplicación en vivo:** [https://url-spam-detector.onrender.com](https://url-spam-detector.onrender.com)

---

## Características

- Visualización completa del pipeline de NLP (tokenización, eliminación de stopwords, lematización, TF-IDF)
- Entrenamiento del modelo SVM con parámetros por defecto y optimizados
- Predicción interactiva de una URL individual
- Predicción por lotes subiendo un archivo CSV
- Guardado y carga del modelo desde disco

---

## Estructura del Proyecto

```
├── app.py                 # Aplicación Streamlit
├── requirements.txt       # Dependencias de Python
├── render.yaml            # Configuración de despliegue en Render
├── setup.sh              # Script de descarga de datos NLTK
├── .gitignore            # Reglas de exclusión de Git
├── 12-nlp-url-spam.ipynb # Notebook Jupyter con el análisis completo
├── data/
│   └── raw/
│       └── url_spam.csv  # Dataset
└── models/
    ├── svm_url_spam_model.pkl        # Modelo SVM entrenado
    └── tfidf_url_spam_vectorizer.pkl # Vectorizador TF-IDF ajustado
```

---

## Ejecutar Localmente

```bash
# Clonar el repositorio
git clone https://github.com/4GeeksAcademy/abgonzalez-machine-learning-streamlit.git
cd abgonzalez-machine-learning-streamlit

# Instalar dependencias
pip install -r requirements.txt

# Descargar datos de NLTK
python -m nltk.downloader stopwords wordnet

# Ejecutar la aplicación
streamlit run app.py
```

La aplicación se abrirá en `http://localhost:8501`.

---

## Desplegar en Render

### Paso 1: Subir el código a GitHub

Asegúrate de que estos archivos estén confirmados y subidos:
- `app.py`
- `requirements.txt`
- `data/raw/url_spam.csv`
- `models/svm_url_spam_model.pkl`
- `models/tfidf_url_spam_vectorizer.pkl`

### Paso 2: Crear un Web Service en Render

1. Ve a [https://render.com](https://render.com) y regístrate o inicia sesión
2. Haz clic en **"New +"** → **"Web Service"**
3. Conecta tu cuenta de GitHub y selecciona el repositorio
4. Configura el servicio:

| Configuración | Valor |
|---------------|-------|
| **Name** | `url-spam-detector` |
| **Runtime** | Python |
| **Branch** | `main` |
| **Build Command** | `pip install -r requirements.txt && python -m nltk.downloader stopwords wordnet` |
| **Start Command** | `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true` |

5. Selecciona el plan **Free** (gratuito)
6. Haz clic en **"Create Web Service"**

### Paso 3: Esperar el despliegue

Render instalará las dependencias, descargará los datos de NLTK e iniciará la aplicación Streamlit. Después de unos minutos, obtendrás una URL pública como:

```
https://url-spam-detector.onrender.com
```

---

## Notas Importantes

- **Comandos en texto plano:** Al pegar los comandos de Build/Start en el dashboard de Render, usa `Cmd+Shift+V` (Mac) o `Ctrl+Shift+V` (Windows) para pegar como texto plano. NO pegues texto con formato desde VS Code.
- **Plan gratuito:** La instancia gratuita de Render se apaga tras un período de inactividad. La primera solicitud después de la inactividad puede tardar 30-60 segundos.
- **Datos NLTK:** El comando de build descarga los paquetes NLTK necesarios (`stopwords` y `wordnet`). Esto se ejecuta una sola vez durante la fase de construcción.

---

## Tecnologías

- Python 3.12
- Streamlit
- scikit-learn (SVM, TF-IDF, GridSearchCV)
- NLTK (stopwords, lematizador WordNet)
- pandas, numpy, matplotlib, seaborn
