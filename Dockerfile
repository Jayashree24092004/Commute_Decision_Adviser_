FROM python:3.10

WORKDIR /code

COPY . /code

RUN pip install --no-cache-dir fastapi uvicorn pydantic

EXPOSE 7860

CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "7860"]