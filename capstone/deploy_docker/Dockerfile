# Use an official colab Python runtime as the base image
FROM python:3.10.12

# Set the working directory in the container
WORKDIR /app/

# Copy the requirements file into the container
COPY requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy data file to run from container
# RUN mkdir /app/data
# COPY data/*.json /app/data

# Copy the rest of the application code into the container
COPY triviaqa_ijson_utils.py /app/
COPY qa_metrics_utils.py /app/
COPY app_ijson_ollama_llm.py /app/
COPY templates/*.html /app/templates/
COPY static/* /app/static/
# COPY eval_llm.py /app/

# Expose a port if needed
EXPOSE 5000

# Command to run when the container starts
CMD ["python", "app_ijson_ollama_llm.py"]

# Open Docker Desktop; 
# Open Powershell, cd to <dir> ; bash ; 
# docker build -t my-llm-flask-app .
# docker run  --env-file .env -d --name my-llm-flask-cont  -p 5000:5000 my-llm-flask-app
# Can look at container log to find url for ngrok; ensure .env has authtoken for ngrok
