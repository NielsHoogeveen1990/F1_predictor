FROM python:3.7

LABEL Author="Niels Hoogeveen"

RUN mkdir /app
WORKDIR /app
ADD . /app/
RUN pip install -r requirements.txt

EXPOSE 6000
ENV FLASK_APP=app_race/app_race.py
ENV FLASK_DEBUG=1

CMD ["flask", "run", "--host=0.0.0.0", "--port=6000"]