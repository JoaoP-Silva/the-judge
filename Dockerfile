FROM ubuntu:22.04

# avoid iterative prompts in installations
ENV DEBIAN_FRONTEND=noninteractive

# install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    python3 \
    python3-pip \
    python3-dev \
    make \
    g++ \
    curl \
    && apt-get clean

# install python dependencies
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip3 install --no-cache-dir -r requirements.txt

# copy project
COPY . /app

# gen c++ binaries
RUN make 

# expose flask port
EXPOSE 5000

# initialize flask server
CMD ["python3", "server/app.py"]
