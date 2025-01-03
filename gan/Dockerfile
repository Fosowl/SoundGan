FROM nvidia/cuda:11.1-base

COPY . .

RUN apt-get update && apt-get install -y \
    && pip3 install --no-cache-dir -r requirements.txt

EXPOSE 5050

# Run ML Microservice
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5050"]