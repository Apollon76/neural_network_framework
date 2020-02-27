FROM ubuntu:18.04

RUN apt update && apt install -y \
	ssh \
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
    liblapack-dev

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
