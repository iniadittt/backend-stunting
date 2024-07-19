FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 \
    APP_HOME=/app
WORKDIR $APP_HOME
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt
COPY . .
EXPOSE 8080
CMD ["python", "main.py"]