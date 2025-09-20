FROM node:20-alpine

WORKDIR /usr/src/app

COPY package*.json ./
RUN npm ci --production

COPY . .

RUN mkdir -p reports

ENV ARTILLERY_SCENARIO=artillery.yml
ENV ARTILLERY_TARGET=

ENTRYPOINT ["/bin/sh","-c","if [ -z \"$ARTILLERY_TARGET\" ]; then echo 'ERROR: set ARTILLERY_TARGET env var'; exit 2; fi; npx artillery run --output=reports/result.json $ARTILLERY_SCENARIO"]
