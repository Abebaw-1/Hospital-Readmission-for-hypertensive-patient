FROM python:3.9.13
ENV PYTHONBEFFERED True
ENV APP_HOME/app
COPY --/app
WORKDIR /app
RUN pip install -r requirements.txt
RUN pip install gunicorn
CMD exec gunicorn --bind :$port --workers 1 --threads 8 --timeout 0 app:app

