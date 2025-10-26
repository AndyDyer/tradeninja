FROM python:3.10-slim

WORKDIR /app

# Install deps (add numpy manually since code imports it but toml misses it)
RUN pip install --no-cache-dir streamlit==1.50.0 pandas==2.3.3 numpy==2.2.6

# Copy app code (assumes csv data files are in trade2/ or handled via upload/volume)
COPY . /app

EXPOSE 8080

# Run streamlit
CMD ["streamlit", "run", "main.py", "--server.port=8080", "--server.address=0.0.0.0"]
