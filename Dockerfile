FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

#uvicorn main:app --port 5000 --ssl-keyfile=./key.pem --ssl-certfile=./cert.pem
