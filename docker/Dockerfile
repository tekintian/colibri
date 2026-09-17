FROM debian:stable-slim 

# We install the necessary packages to download and compile the code
RUN apt-get update && \
        apt-get install -y locales git build-essential python3 python3-pip && \
        rm -rf /var/lib/apt/lists/* && \
        mkdir /app

# Let's clone the repository. Pin the checkout so builds are reproducible
# (M-3): a floating clone of the default branch makes "docker build" mean
# "whatever upstream looks like today". Override at build time to track main:
#   docker build --build-arg COLIBRI_REF=main -f docker/Dockerfile .
ARG COLIBRI_REF=v1.11.0
WORKDIR /app
RUN git clone --depth 1 --branch ${COLIBRI_REF} https://github.com/JustVugg/colibri.git .
WORKDIR /app/c
# It's time to compile
RUN ./setup.sh && ./coli build

# Drop to an unprivileged user for everything after the build (M-3), matching
# docker/Dockerfile.slim: the launcher, gateway and engine never need root,
# and a root-owned /app only ever helped a compromised process.
RUN useradd -u 1000 -m colibri && chown -R colibri:colibri /app
USER colibri
ENV HOME=/home/colibri
