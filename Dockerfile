FROM alpine

# Installs latest Chromium (100) package.
RUN apk add --no-cache \
      chromium \
      nss \
      freetype \
      harfbuzz \
      ca-certificates \
      ttf-freefont \
      nodejs \
      yarn

RUN apk add --no-cache texlive
# Tell Puppeteer to skip installing Chrome. We'll be using the installed package.
ENV PUPPETEER_EXECUTABLE_PATH=/usr/bin/chromium-browser

# Puppeteer v13.5.0 works with Chromium 100.
RUN yarn add puppeteer@13.5.0 && mkdir -p /app

WORKDIR /app
COPY yarn.lock ./
COPY package.json ./
COPY dependencies/ ./dependencies
RUN ls -alh #bump
RUN yarn install

ENV PUPPETEER_SKIP_CHROMIUM_DOWNLOAD true
ENV RESUME_PUPPETEER_NO_SANDBOX true
COPY . .

# Add user so we don't need --no-sandbox.
RUN addgroup -S pptruser && adduser -S -G pptruser pptruser \
    && mkdir -p /home/pptruser/Downloads /app \
    && chown -R pptruser:pptruser /home/pptruser \
    && chown -R pptruser:pptruser /app

USER pptruser
# Run everything after as non-privileged user.
ENTRYPOINT [ "yarn", "run", "resume" ]