FROM python:3.12-slim as base

RUN apt-get update && apt-get install -y poppler-utils && rm -rf /var/lib/apt/lists/*

ENV  POETRY_VERSION=1.6.1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PYSETUP_PATH="/opt/pysetup" \
    VENV_PATH="/opt/pysetup/.venv"
ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"


FROM base as builder
RUN --mount=type=cache,target=/root/.cache \
    pip install "poetry==$POETRY_VERSION"
WORKDIR $PYSETUP_PATH
COPY ./poetry.lock ./pyproject.toml ./
RUN --mount=type=cache,target=$POETRY_HOME/pypoetry/cache \
    poetry install --no-dev


FROM base as production
ENV FASTAPI_ENV=production
COPY --from=builder $VENV_PATH $VENV_PATH
COPY ./colpali_search /colpali_search
COPY .env /colpali_search/
COPY alembic.ini /colpali_search/
COPY alembic /colpali_search/alembic
WORKDIR /colpali_search
EXPOSE 8000
# Remove the following when running docker-compose.yml
CMD ["fastapi", "run", "app.py", "--port", "8000", "--workers", "1"]