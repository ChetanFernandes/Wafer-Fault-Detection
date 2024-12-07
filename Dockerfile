FROM python:3.10-slim-buster

WORKDIR /app

COPY . /app

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y build-essential libssl-dev libffi-dev python3-dev

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python3", "app.py"]