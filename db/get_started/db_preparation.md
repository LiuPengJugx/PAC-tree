```
<!-- 创建容器 -->
docker run -d \
-p 5432:5432 \
-e POSTGRES_USER=postgres \
-e POSTGRES_PASSWORD=postgres \
-v /Users/liupengju/docker-home/pg:/var/lib/postgresql/data \
--name pg \
--restart always \
docker.io/postgres:13-alpine3.21



<!-- 进入容器 -->
docker exec -it a964ac2ceb53 /bin/bash 

psql -U postgres



<!-- 下载编译pg_hint_plan插件 -->

wget https://github.com/ossc-db/pg_hint_plan/archive/refs/tags/REL13_1_3_9.tar.gz

tar xzvf REL13_1_3_9.tar.gz

cd pg_hint_plan-REL13_1_3_9

apk add \
  gcc \
  libc-dev \
  make \
  llvm19 \
  clang19

make install

<!-- 安装 pg_hint_plan和pageinspect插件-->

psql -U postgres -c "CREATE EXTENSION pg_hint_plan;"

psql -U postgres -c "CREATE EXTENSION pageinspect;"
```