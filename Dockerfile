FROM ubuntu:jammy as build

COPY requirements.txt /requirements.txt

RUN apt update && apt install -y \ 
    python3-pip libgl1 libglib2.0-0 x11-apps \
    && pip install -r requirements.txt

## TODO: 
    # Clone, build, and install pangolin
    # Extract images from tar file
    # Install any dependencies

COPY vo.py /app/vo.py

RUN chmod +x /app/vo.py

### Final stage build
FROM scratch

COPY --from=build / /

WORKDIR /app

CMD ["./vo.py"]