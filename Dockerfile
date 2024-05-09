FROM ubuntu:jammy as build

COPY requirements.txt /requirements.txt

RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y \ 
    python3-pip git x11-apps cmake python3-tk \
    libgl1-mesa-dev libglew-dev libeigen3-dev \
    && pip install -r requirements.txt

WORKDIR /app

# Build my forked version of pangolin
RUN git clone https://github.com/gilbertgonz/pangolin.git \
    && cd pangolin && mkdir build && cd build \
    && cmake .. && make -j8 \
    && cd .. \
    && python3 setup.py install || true

COPY vo.py  /app/vo.py
COPY assets /app/assets
COPY libs   /app/libs

RUN chmod +x /app/vo.py

### Final stage build
FROM scratch

COPY --from=build / /

WORKDIR /app

CMD ["./vo.py"]