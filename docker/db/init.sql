-- Create databases
CREATE DATABASE imdb;
CREATE DATABASE tpch;
CREATE DATABASE tpcds;

-- Create users
CREATE USER spark WITH PASSWORD 'spark';
GRANT ALL PRIVILEGES ON DATABASE imdb TO spark;
GRANT ALL PRIVILEGES ON DATABASE tpch TO spark;
GRANT ALL PRIVILEGES ON DATABASE tpcds TO spark;

-- Allow remote connections
ALTER SYSTEM SET listen_addresses = '*';
SELECT pg_reload_conf();