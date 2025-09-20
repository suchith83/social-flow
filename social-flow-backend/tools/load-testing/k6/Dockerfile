FROM grafana/k6:latest

WORKDIR /src
COPY . .

ENV K6_TARGET=https://api.example.com

ENTRYPOINT ["k6"]
