FROM node:20-alpine

WORKDIR /usr/src/app

COPY package*.json ./
RUN npm ci --production

COPY . .

RUN mkdir -p reports

CMD ["node", "scripts/run_experiment.js", "experiments/cpu_stress.json"]
