FROM python:3.10-slim

WORKDIR /app

# Copiar los archivos de dependencias y la aplicación
COPY requirements.txt .
COPY app.py .

# Instalar las librerías de Python directamente
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto por defecto de Streamlit
EXPOSE 8501

# Comando para ejecutar la aplicación
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
