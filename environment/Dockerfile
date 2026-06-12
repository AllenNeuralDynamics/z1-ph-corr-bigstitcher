# hash:sha256:3a36642cc591313c46df25d8c09e4663cfc9579b6a1a05215507f9ea51529dd2
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

LABEL maintainer="Sean Fite <sean.fite@alleninstitute.org>"

ARG DEBIAN_FRONTEND=noninteractive

ARG AWS_ACCESS_KEY_ID
ARG AWS_DEFAULT_REGION
ARG AWS_SECRET_ACCESS_KEY

ENV AWS_REGION=us-west-2
ENV JAVA_OPTS="-Djava.awt.headless=true"
ENV HOME="/root"
ENV BIGSTITCHER_HOME=/home/BigStitcher-Spark
ENV N5_AWS_FOLDER_HOME=/home/n5-aws-s3
ENV CLASSPATH="$BIGSTITCHER_HOME/target/*:$N5_AWS_FOLDER_HOME/target/*"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget \
        git \
        libblosc-dev=1.21.1+ds2-2 \
        maven=3.6.3-5 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR $HOME

RUN wget -q https://cdn.azul.com/zulu/bin/zulu8.80.0.17-ca-fx-jdk8.0.422-linux_x64.tar.gz && \
    tar -xzf zulu8.80.0.17-ca-fx-jdk8.0.422-linux_x64.tar.gz && \
    rm zulu8.80.0.17-ca-fx-jdk8.0.422-linux_x64.tar.gz && \
    mv zulu8.80.0.17-ca-fx-jdk8.0.422-linux_x64 /opt/java

ENV JAVA_HOME="/opt/java"
ENV PATH="$JAVA_HOME/bin:$PATH"

WORKDIR /.code-server

RUN wget -qO code-server.tar.gz https://github.com/coder/code-server/releases/download/v4.9.0/code-server-4.9.0-linux-amd64.tar.gz && \
    tar -xvf code-server.tar.gz --strip-components=1 && \
    rm code-server.tar.gz && \
    ln -s /.code-server/bin/code-server /usr/bin/code-server

RUN pip3 install --no-cache-dir \
    dask==2024.1.1 \
    dask-image==2023.8.1 \
    numpy==1.26.3 \
    pathlib==1.0.1 \
    psutil==5.9.5 \
    regex==2023.10.3 \
    toml==0.10.2 \
    zarr==2.16.1 \
    natsort==8.4.0 \
    matplotlib==3.9.2 \
    aind-data-schema==1.4.0 \
    numcodecs==0.13.1 \
    boto3 \
    xmltodict \
    s3fs

COPY postInstall /
RUN /bin/bash /postInstall

WORKDIR $HOME

