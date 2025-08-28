FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install OS build dependencies
RUN apt-get update && \
    apt-get install -y \ 
	python3 \
	python3-pip \
	python3-dev \
	python3-distutils \
	git \
	software-properties-common \
	python3-launchpadlib \
	gnupg2 \
	ca-certificates \
	build-essential \
	python3-setuptools \
	make \
	cmake \
	ffmpeg \
	libavcodec-dev \
	libavfilter-dev \
	libavformat-dev \
	libavutil-dev \
	tesseract-ocr

# Install numpy to avoid 1.18.2 installation issues and to build decord
RUN pip install numpy 

# Clone decord
WORKDIR /opt
RUN git clone --recursive https://github.com/dmlc/decord

# Clone and build decord
WORKDIR /opt/decord
RUN mkdir build && cd build && \
	cmake .. -DUSE_CUDA=0 -DCMAKE_BUILD_TYPE=Release && \
	make 
RUN cp build/libdecord.so python/decord/
WORKDIR /opt/decord/python
RUN pip install .

# Implicitly required to load SVM
RUN pip install scikit-learn==1.0.2 

# Final workdir
WORKDIR /workspace
