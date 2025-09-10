FROM golang:1.21-alpine AS builder
ADD . /go/src/user-service
WORKDIR /go/src/user-service
RUN go build -o /user-service ./cmd/main.go

FROM alpine:3.18
COPY --from=builder /user-service /
CMD ["/user-service"]
