docker pull bitnami/spark

docker run -itd -p 8080:8080 -p 7077:7077 --name=spark -e SPARK_MODE=master bitnami/spark:latest

docker-compose up -d

docker exec -it -u root 容器ID sh

