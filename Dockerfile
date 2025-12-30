FROM rust:1.92.0 as builder

RUN apt-get update && apt-get install -y pkg-config libssl-dev protobuf-compiler libasound2-dev

WORKDIR /usr/src/app
COPY . .

RUN cargo install --path .

CMD ["llm"]