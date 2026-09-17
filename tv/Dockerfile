# Dockerized TradingView fetcher. The @mathieuc/tradingview library is installed INSIDE the image only —
# it never runs on the host (per project constraint). Build: docker compose build tv
FROM node:20-alpine
WORKDIR /app
COPY package.json ./
RUN npm install --omit=dev
COPY . .
# ENTRYPOINT is node so `docker compose run --rm tv <script.js> <args...>` just works.
ENTRYPOINT ["node"]
CMD ["tv_fetch.js"]
