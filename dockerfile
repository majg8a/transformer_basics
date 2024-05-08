FROM python:3.12.3-bullseye
RUN pip install --upgrade pip
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
CMD python script.py 