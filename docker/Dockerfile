FROM ubuntu:18.04

RUN apt-get update && apt-get install -y \
	ssh \
	git \
	make \
	cmake \
	clang-8 \
	clang-format-8 \
	clang-tidy-8 \
	ca-certificates \
	openssh-server \
	rsync \
	gdb \
	wget \
	autoconf \
    libopenblas-dev \
    liblapack-dev \
    libgtest-dev \
    python3 \
    python3-pip \
    automake \
    libtool \
    g++ \
    unzip \
    libjpeg-dev \
    libpng-dev \
    libhdf5-dev \
    libboost-all-dev \
    libprotobuf-dev \
    protobuf-compiler

# Armadillo:
ENV ARMA_VERSION armadillo-9.850.1
RUN mkdir -p /tmp/arma
WORKDIR /tmp/arma
RUN wget http://sourceforge.net/projects/arma/files/$ARMA_VERSION.tar.xz
RUN tar -xpJf $ARMA_VERSION.tar.xz
WORKDIR /tmp/arma/$ARMA_VERSION
ENV CXX="/usr/bin/clang++-8"
ENV CC="/usr/bin/clang-8"
RUN cmake .
RUN make && make install
WORKDIR /
RUN rm -rf /tmp/arma

# Glog:
RUN git clone https://github.com/google/glog.git /tmp/glog
WORKDIR /tmp/glog
RUN cmake -H. -Bbuild
RUN cmake --build build
RUN cmake --build build --target install

# Gtest:
WORKDIR /usr/src/gtest
RUN cmake CMakeLists.txt
RUN make
RUN cp libgtest.a /usr/lib
RUN cp libgtest_main.a /usr/lib

# Csv:
RUN git clone https://github.com/ben-strasser/fast-cpp-csv-parser.git /tmp/csv
RUN mkdir /usr/include/csv && cp /tmp/csv/csv.h /usr/include/csv

# Argument Parser
RUN git clone https://github.com/jarro2783/cxxopts.git /tmp/cxxopts
RUN cp /tmp/cxxopts/include/cxxopts.hpp /usr/include/cxxopts.hpp

# JSON
RUN wget https://github.com/nlohmann/json/releases/download/v3.7.3/json.hpp
RUN mkdir -p /usr/include/nlohmann && mv json.hpp /usr/include/nlohmann/json.hpp

# cereal
RUN git clone https://github.com/USCiLab/cereal.git /tmp/cereal
RUN mkdir -p /usr/include/ && mv /tmp/cereal/include/cereal  /usr/include/

# CImg
RUN mkdir -p /tmp/CImg
WORKDIR /tmp/CImg
RUN wget https://github.com/dtschump/CImg/archive/v.2.8.4.zip
RUN unzip v.2.8.4.zip && cp CImg-v.2.8.4/CImg.h /usr/include/

# pbar
RUN mkdir -p /tmp/pbar
WORKDIR /tmp/pbar
RUN git clone https://github.com/Jvanrhijn/pbar.git && cp pbar/pbar.h /usr/include

# HighFive
RUN mkdir -p /tmp/HighFive
WORKDIR /tmp/HighFive
RUN wget https://github.com/BlueBrain/HighFive/archive/v2.2.tar.gz && tar -xzf v2.2.tar.gz && rm v2.2.tar.gz
WORKDIR /tmp/HighFive/HighFive-2.2
RUN cmake -DHIGHFIVE_EXAMPLES=OFF && make install

# sshd:
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN echo "StrictHostKeyChecking=no" >> /etc/ssh/ssh_config
EXPOSE 22
RUN groupadd -g 10000 user
RUN useradd -u 10000 -g 10000 -m user && yes password | passwd user
RUN mkdir /var/run/sshd

RUN echo 'CXX="/usr/bin/clang++-8"' >> /etc/environment
RUN echo 'CC="/usr/bin/clang-8"' >> /etc/environment

# entry point:
COPY docker-entry-point.sh /
ENTRYPOINT ["/docker-entry-point.sh"]
