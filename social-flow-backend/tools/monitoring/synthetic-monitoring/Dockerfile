FROM mcr.microsoft.com/playwright:v1.45.0-focal

WORKDIR /usr/src/app

COPY package*.json ./
RUN npm ci --production

COPY . .

RUN mkdir -p reports

CMD ["node", "scripts/run_check.js", "scenarios/api_check.js"]
