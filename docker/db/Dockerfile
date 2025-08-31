FROM postgres:15

# Install additional extensions if needed
RUN apt-get update && apt-get install -y \
    postgresql-contrib \
    && rm -rf /var/lib/apt/lists/*

# Copy initialization scripts
COPY init.sql /docker-entrypoint-initdb.d/

# Set custom configuration
COPY postgresql.conf /etc/postgresql/postgresql.conf

CMD ["postgres", "-c", "config_file=/etc/postgresql/postgresql.conf"]