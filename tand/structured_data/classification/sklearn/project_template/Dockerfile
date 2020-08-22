FROM python:3.8

WORKDIR "/app"

COPY . .

RUN ["pip", "install", "python-dotenv"]

RUN ["pip", "install", "-r", "requirements.txt"]

EXPOSE 8000

ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]