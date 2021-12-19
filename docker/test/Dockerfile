FROM python:3.9-buster

WORKDIR /stage

COPY . .

RUN pip install -U pip && pip install poetry==1.1.2
RUN poetry install  # You don't need to specify `--dev` because the command installs dev dependencies by default
