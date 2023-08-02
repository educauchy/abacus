FROM jupyter/datascience-notebook:latest
MAINTAINER Vadim Glukhov <educauchy@gmail.com>

EXPOSE 8889

COPY ./abacus /usr/local/abacus/abacus
COPY ./examples /usr/local/abacus/examples
COPY ./requirements.txt /usr/local/abacus/requirements.txt
COPY ./main.py /usr/local/abacus/main.py
WORKDIR /usr/local/abacus

RUN pip install -r requirements.txt

CMD ["python", "main.py"]