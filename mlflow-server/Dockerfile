FROM python:3.8-slim-buster
RUN pip install mlflow boto3 pymysql psycopg2-binary
WORKDIR /home/mlflow

EXPOSE 5000

ENTRYPOINT ["mlflow", "server"]
