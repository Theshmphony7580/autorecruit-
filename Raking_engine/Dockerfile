FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install streamlit

# Copy the entire Raking_engine directory
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Run the Streamlit app with a 1GB upload limit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.maxUploadSize=1000"]
