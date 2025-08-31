FROM bitnami/spark:3.4.0

# Install additional packages
USER root
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --no-cache-dir \
    pyspark==3.4.0 \
    pandas \
    numpy \
    psycopg2-binary

USER 1001

# Set environment variables
ENV SPARK_MODE=master
ENV SPARK_RPC_AUTHENTICATION_ENABLED=no
ENV SPARK_RPC_ENCRYPTION_ENABLED=no
ENV SPARK_LOCAL_STORAGE_ENCRYPTION_ENABLED=no
ENV SPARK_SSL_ENABLED=no

EXPOSE 8080 7077 4040