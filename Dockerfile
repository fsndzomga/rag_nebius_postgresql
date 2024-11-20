# Use Ubuntu 20.04 LTS as a base image
FROM ubuntu:20.04

# Prevent apt from asking questions when installing packages
ARG DEBIAN_FRONTEND=noninteractive

# Set the working directory in the container
WORKDIR /app

# Install PostgreSQL, Python, Tesseract-OCR, and other necessary packages in one RUN command
RUN apt-get update && apt-get install -y \
    postgresql postgresql-contrib \
    python3 python3-pip python3-dev \
    gcc libpq-dev \
    poppler-utils \
    tesseract-ocr \
    libtesseract-dev \
    git \
    curl \
    && sed -i 's/local   all             postgres                                peer/local   all             postgres                                trust/' /etc/postgresql/12/main/pg_hba.conf \
    && sed -i 's/local   all             all                                     peer/local   all             all                                     trust/' /etc/postgresql/12/main/pg_hba.conf \
    && echo "host all  all    0.0.0.0/0  trust" >> /etc/postgresql/12/main/pg_hba.conf \
    && echo "listen_addresses='*'" >> /etc/postgresql/12/main/postgresql.conf \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --no-cache-dir fastapi uvicorn sqlalchemy psycopg2-binary python-dotenv sqlalchemy-utils pgvector pydantic PyPDF2 pytesseract Pillow PyMuPDF nltk openai python-multipart

# Install PostgreSQL development files for version 12
RUN apt-get update && apt-get install -y postgresql-server-dev-12

# Set the path to pg_config
ENV PG_CONFIG=/usr/bin/pg_config

# Clone the pgvector repository and install the extension
RUN cd /tmp && \
    git clone --branch v0.6.2 https://github.com/pgvector/pgvector.git && \
    cd pgvector && \
    make && \
    make install

# Copy the current directory contents into the container at /app
COPY . /app

# Expose the port your app runs on
EXPOSE 80
