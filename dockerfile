FROM python:3.8.10-slim
WORKDIR /usr/src/app
COPY requirements.txt .
RUN apt-get -y update && apt-get -y install --reinstall build-essential
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 80 84
# CMD flask run
COPY . /usr/src/app
CMD gunicorn --bind 0.0.0.0:8000 app:app --timeout 90