FROM python:3.9-slim
WORKDIR /docker_env

COPY requirements.txt .

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install --no-cache-dir -r requirements.txt

COPY ./analysis ./analysis
COPY ./medi-model ./medi-model

CMD ["python3", "./medi-model/model.py"]