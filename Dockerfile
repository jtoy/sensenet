# A Dockerfile that sets up a full SenseNet install
FROM ubuntu:14.04

RUN apt-get update \
    && apt-get install -y libav-tools \
    python-numpy \
    python-scipy \
    python-pyglet \
    python-setuptools \
    libpq-dev \
    libjpeg-dev \
    curl \
    cmake \
    swig \
    freeglut3 \
    python-opengl \
    libboost-all-dev \
    libglu1-mesa \
    libglu1-mesa-dev \
    libsdl2-2.0-0\
    libgles2-mesa-dev \
    libsdl2-dev \
    wget \
    unzip \
    git \
    xserver-xorg-input-void \
    xserver-xorg-video-dummy \
    python-gtkglext1 \
    xpra \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && easy_install pip

WORKDIR /usr/local/sensenet
RUN mkdir -p sensenet && touch sensenet/__init__.py
COPY ./sensenet/version.py ./sensenet
COPY ./requirements.txt .
COPY ./setup.py .
RUN pip install -e .[all]

# Finally, upload our actual code!
COPY . /usr/local/sensenet

WORKDIR /root
