# Image for NVIDIA Orin NX - Jetpack 5.1 [L4T 35.2.1]
FROM nvcr.io/nvidia/l4t-ml:r35.2.1-py3 

WORKDIR /app

RUN python3 -m pip install --upgrade pip setuptools wheel

RUN python3 -m pip install keras-ocr

RUN python3 -m pip install "opencv-python-headless<4.3" easyocr loguru
RUN python3 -m pip install ultralytics --no-deps tqdm onnxruntime


COPY sampleset/ /app/sampleset/
COPY ysco.py /app/
COPY ysco_config.json /app/
COPY models/ /app/models/

# CMD ["python3", "ysco.py"]

