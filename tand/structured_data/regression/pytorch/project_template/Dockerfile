FROM python:3.8

WORKDIR "/app"

COPY . .

RUN ["pip", "install", "python-dotenv", "--no-cache-dir"]

RUN ["pip", "install", "-r", "requirements.txt", "--no-cache-dir"]

EXPOSE 8000

ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]