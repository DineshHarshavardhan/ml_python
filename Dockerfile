# Build an image that can do training and inference in SageMaker
# This is a Python 3 image that uses the nginx, gunicorn, flask stack
# for serving inferences in a stable way.

FROM ubuntu:18.04

MAINTAINER Leongn

# 1. Define the packages required in our environment. 


RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         python3 \
         nginx \
         ca-certificates \
         python3-pip \
         ipython3 \
         build-essential \
         python3-dev \
         python3-setuptools \
    && rm -rf /var/lib/apt/lists/*
    
    
RUN apt-get -y update && apt-get install -y --no-install-recommends \
         nvidia-cuda-toolkit \
         nvidia-modprobe \
         cuda-10.2 \
    && rm -rf /var/lib/apt/lists/*
 
 
 
RUN wget https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py && \
    pip3 install setuptools==49.3.0 numpy==1.18.5 scipy==1.4.1 sklearn scikit-learn==0.22.2.post1 tensorflow==2.2  sentence_transformers boto3 pandas torch torchvision transformers sentence-transformers flask gevent gunicorn && \
        rm -rf /root/.cache
    
    

RUN wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda_8.0.61_375.26_linux-run

RUN apt-get -y update

RUN apt-get -y install gnupg


# 2. Here we define all python packages we want to include in our environment.
# Pip leaves the install caches populated which uses a significant amount of space. 
# These optimizations save a fair amount of space in the image, which reduces start up time.

# 3. Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# 4. Define the folder (sentiment_analysis) where our inference code is located
COPY parkers /opt/program
WORKDIR /opt/program

