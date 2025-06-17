FROM swaggerapi/swagger-ui:v4.18.2 AS swagger-ui
FROM python:3.12-slim

#–– Environment variables ––
ENV POETRY_VENV=/app/.venv
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

#–– 1) Install ffmpeg (and clean up apt lists) ––
RUN apt-get update -qq && \
    apt-get install -qq --no-install-recommends \
      ffmpeg \
    && rm -rf /var/lib/apt/lists/*

#–– 2) Create a venv at /app/.venv and install Poetry 2.1.3 into it ––
RUN python3 -m venv $POETRY_VENV && \
    $POETRY_VENV/bin/pip install -U pip "setuptools<81" && \
    $POETRY_VENV/bin/pip install poetry==2.1.3

# Add Poetry’s bin‐folder to PATH so "poetry" is available globally
ENV PATH="${PATH}:${POETRY_VENV}/bin"

WORKDIR /app

#–– 3) Copy lockfiles, configure Poetry, install dependencies without your source code ––
COPY poetry.lock pyproject.toml ./

RUN poetry config virtualenvs.in-project true && \
    poetry install --no-root

#–– 4) Copy application code and swagger‐ui assets ––
COPY . .
COPY --from=swagger-ui /usr/share/nginx/html/swagger-ui.css    swagger-ui-assets/swagger-ui.css
COPY --from=swagger-ui /usr/share/nginx/html/swagger-ui-bundle.js swagger-ui-assets/swagger-ui-bundle.js

#–– 5) Install any remaining dev/runtime deps (if your pyproject.toml needs it) ––
RUN poetry --version
RUN poetry install

#–– 6) Start Gunicorn as before ––
ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9000", "--workers", "1", "--timeout", "0", "app.webservice:app", "-k", "uvicorn.workers.UvicornWorker"]
CMD []
