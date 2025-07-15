FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libfreetype6-dev libpng-dev libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch==2.0.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "models/train.py"]