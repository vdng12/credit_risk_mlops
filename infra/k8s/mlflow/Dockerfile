FROM python:3.13-slim

RUN pip install --no-cache-dir \
    mlflow>=2.19 \
    boto3>=1.36 \
    psycopg2-binary>=2.9

EXPOSE 5000

ENTRYPOINT ["mlflow"]
CMD ["server", "--help"]