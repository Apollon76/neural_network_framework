FROM ubuntu:18.04

RUN apt update && apt install -y \
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
    libgtest-dev

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
COPY development.sh /
COPY run.sh /
COPY test.sh /
ENTRYPOINT ["/development.sh"]
