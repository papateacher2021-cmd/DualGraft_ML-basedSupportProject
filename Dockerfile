# Utiliza una imagen oficial ligera de Python
FROM python:3.10-slim

# Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copiar los ficheros de requisitos e instalar dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente de la aplicación
COPY . .

# Exponer el puerto por defecto que utiliza Streamlit
EXPOSE 8085

# Configurar variables de entorno para Streamlit en modo headless
ENV STREAMLIT_SERVER_PORT=8085
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_SERVER_ADDRESS=0.0.0.0

# Comando para ejecutar la aplicación al iniciar el contenedor
CMD ["streamlit", "run", "app.py"]
