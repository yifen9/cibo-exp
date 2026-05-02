FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN --mount=type=cache,target=/var/cache/apt --mount=type=cache,target=/var/lib/apt \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      bash \
      ca-certificates \
      curl \
      git \
      build-essential \
      pkg-config \
      libssl-dev \
      python3 \
      python3-venv \
      python3-dev \
    && rm -rf /var/lib/apt/lists/*

ENV UV_CACHE_DIR=/opt/uv-cache
RUN mkdir -p /opt/uv-cache && chmod -R 777 /opt/uv-cache
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN install -m 0755 /root/.local/bin/uv /usr/local/bin/uv

ENV RUSTUP_HOME=/opt/rustup
ENV CARGO_HOME=/opt/cargo
ENV PATH=/opt/cargo/bin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN mkdir -p /opt/rustup /opt/cargo && chmod -R 777 /opt/rustup /opt/cargo
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
RUN rustup component add rustfmt clippy
RUN cargo install just

WORKDIR /workspace
RUN git config --system --add safe.directory /workspace

CMD ["bash","-lc","exec sleep infinity"]
