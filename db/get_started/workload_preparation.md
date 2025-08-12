# IMDB
## get data

```
wget http://homepages.cwi.nl/~boncz/job/imdb.tgz
tar -xvzf imdb.tgz -C imdb-datasets-ftp/
rm -f imdb.tgz
```

```
psql -c "DROP DATABASE IF EXISTS imdb"

psql -c "CREATE DATABASE imdb"
```

```sql
CREATE TABLE aka_name (  
id integer NOT NULL PRIMARY KEY,  
person_id integer NOT NULL,  
name character varying,  
imdb_index character varying(3),  
name_pcode_cf character varying(11),  
name_pcode_nf character varying(11),  
surname_pcode character varying(11),  
md5sum character varying(65)  
);  
  
CREATE TABLE aka_title (  
id integer NOT NULL PRIMARY KEY,  
movie_id integer NOT NULL,  
title character varying,  
imdb_index character varying(4),  
kind_id integer NOT NULL,  
production_year integer,  
phonetic_code character varying(5),  
episode_of_id integer,  
season_nr integer,  
episode_nr integer,  
note character varying(72),  
md5sum character varying(32)  
);  
  
CREATE TABLE cast_info (  
id integer NOT NULL PRIMARY KEY,  
person_id integer NOT NULL,  
movie_id integer NOT NULL,  
person_role_id integer,  
note character varying,  
nr_order integer,  
role_id integer NOT NULL  
);  
  
CREATE TABLE char_name (  
id integer NOT NULL PRIMARY KEY,  
name character varying NOT NULL,  
imdb_index character varying(2),  
imdb_id integer,  
name_pcode_nf character varying(5),  
surname_pcode character varying(5),  
md5sum character varying(32)  
);  
  
CREATE TABLE comp_cast_type (  
id integer NOT NULL PRIMARY KEY,  
kind character varying(32) NOT NULL  
);  
  
CREATE TABLE company_name (  
id integer NOT NULL PRIMARY KEY,  
name character varying NOT NULL,  
country_code character varying(6),  
imdb_id integer,  
name_pcode_nf character varying(5),  
name_pcode_sf character varying(5),  
md5sum character varying(32)  
);  
  
CREATE TABLE company_type (  
id integer NOT NULL PRIMARY KEY,  
kind character varying(32)  
);  
  
CREATE TABLE complete_cast (  
id integer NOT NULL PRIMARY KEY,  
movie_id integer,  
subject_id integer NOT NULL,  
status_id integer NOT NULL  
);  
  
CREATE TABLE info_type (  
id integer NOT NULL PRIMARY KEY,  
info character varying(32) NOT NULL  
);  
  
CREATE TABLE keyword (  
id integer NOT NULL PRIMARY KEY,  
keyword character varying NOT NULL,  
phonetic_code character varying(5)  
);  
  
CREATE TABLE kind_type (  
id integer NOT NULL PRIMARY KEY,  
kind character varying(15)  
);  
  
CREATE TABLE link_type (  
id integer NOT NULL PRIMARY KEY,  
link character varying(32) NOT NULL  
);  
  
CREATE TABLE movie_companies (  
id integer NOT NULL PRIMARY KEY,  
movie_id integer NOT NULL,  
company_id integer NOT NULL,  
company_type_id integer NOT NULL,  
note character varying  
);  
  
CREATE TABLE movie_info_idx (  
id integer NOT NULL PRIMARY KEY,  
movie_id integer NOT NULL,  
info_type_id integer NOT NULL,  
info character varying NOT NULL,  
note character varying(1)  
);  
  
CREATE TABLE movie_keyword (  
id integer NOT NULL PRIMARY KEY,  
movie_id integer NOT NULL,  
keyword_id integer NOT NULL  
);  
  
CREATE TABLE movie_link (  
id integer NOT NULL PRIMARY KEY,  
movie_id integer NOT NULL,  
linked_movie_id integer NOT NULL,  
link_type_id integer NOT NULL  
);  
  
CREATE TABLE name (  
id integer NOT NULL PRIMARY KEY,  
name character varying NOT NULL,  
imdb_index character varying(9),  
imdb_id integer,  
gender character varying(1),  
name_pcode_cf character varying(5),  
name_pcode_nf character varying(5),  
surname_pcode character varying(5),  
md5sum character varying(32)  
);  
  
CREATE TABLE role_type (  
id integer NOT NULL PRIMARY KEY,  
role character varying(32) NOT NULL  
);  
  
CREATE TABLE title (  
id integer NOT NULL PRIMARY KEY,  
title character varying NOT NULL,  
imdb_index character varying(5),  
kind_id integer NOT NULL,  
production_year integer,  
imdb_id integer,  
phonetic_code character varying(5),  
episode_of_id integer,  
season_nr integer,  
episode_nr integer,  
series_years character varying(49),  
md5sum character varying(32)  
);  
  
CREATE TABLE movie_info (  
id integer NOT NULL PRIMARY KEY,  
movie_id integer NOT NULL,  
info_type_id integer NOT NULL,  
info character varying NOT NULL,  
note character varying  
);  
  
CREATE TABLE person_info (  
id integer NOT NULL PRIMARY KEY,  
person_id integer NOT NULL,  
info_type_id integer NOT NULL,  
info character varying NOT NULL,  
note character varying  
);
```

## import data

```sql
copy aka_name from '/imdb/aka_name.csv' WITH(format csv,DELIMITER ',',NULL '',quote '"',escape '\');

COPY aka_title FROM '/imdb/aka_title.csv' WITH(format csv,DELIMITER ',',NULL '',quote '"',escape '\');

COPY cast_info FROM '/imdb/cast_info.csv' WITH(format csv,DELIMITER ',',NULL '',quote '"',escape '\');

COPY char_name FROM '/imdb/char_name.csv' WITH(format csv,DELIMITER ',',NULL '',quote '"',escape '\');

COPY comp_cast_type FROM '/imdb/comp_cast_type.csv' WITH(format csv,DELIMITER ',',NULL '',quote '"',escape '\');

COPY company_name FROM '/imdb/company_name.csv' WITH(format csv,DELIMITER ',',NULL '',quote '"',escape '\');

COPY company_type FROM '/imdb/company_type.csv' WITH(format csv,DELIMITER ',',NULL '',quote '"',escape '\');

COPY complete_cast FROM '/imdb/complete_cast.csv' WITH(format csv,DELIMITER ',',NULL '',quote '"',escape '\');

COPY info_type FROM '/imdb/info_type.csv' WITH(format csv,DELIMITER ',',NULL '',quote '"',escape '\');

COPY keyword FROM '/imdb/keyword.csv' WITH(format csv,DELIMITER ',',NULL '',quote '"',escape '\');

COPY kind_type FROM '/imdb/kind_type.csv' WITH(format csv,DELIMITER ',',NULL '',quote '"',escape '\');

COPY link_type FROM '/imdb/link_type.csv' WITH(format csv,DELIMITER ',',NULL '',quote '"',escape '\');

COPY movie_companies FROM '/imdb/movie_companies.csv' WITH(format csv,DELIMITER ',',NULL '',quote '"',escape '\');

COPY movie_info FROM '/imdb/movie_info.csv' WITH(format csv,DELIMITER ',',NULL '',quote '"',escape '\');

COPY movie_info_idx FROM '/imdb/movie_info_idx.csv' WITH(format csv,DELIMITER ',',NULL '',quote '"',escape '\');

COPY movie_keyword FROM '/imdb/movie_keyword.csv' WITH(format csv,DELIMITER ',',NULL '',quote '"',escape '\');

COPY movie_link FROM '/imdb/movie_link.csv' WITH(format csv,DELIMITER ',',NULL '',quote '"',escape '\');

COPY name FROM '/imdb/name.csv' WITH(format csv,DELIMITER ',',NULL '',quote '"',escape '\');

COPY person_info FROM '/imdb/person_info.csv' WITH(format csv,DELIMITER ',',NULL '',quote '"',escape '\');

COPY role_type FROM '/imdb/role_type.csv' WITH(format csv,DELIMITER ',',NULL '',quote '"',escape '\');

COPY title FROM '/imdb/title.csv' WITH(format csv,DELIMITER ',',NULL '',quote '"',escape '\');


```

## Create Index
```sql
create index company_id_movie_companies on movie_companies(company_id);

create index company_type_id_movie_companies on movie_companies(company_type_id);

create index info_type_id_movie_info_idx on movie_info_idx(info_type_id);

create index info_type_id_movie_info on movie_info(info_type_id);

create index info_type_id_person_info on person_info(info_type_id);

create index keyword_id_movie_keyword on movie_keyword(keyword_id);

create index kind_id_aka_title on aka_title(kind_id);

create index kind_id_title on title(kind_id);

create index linked_movie_id_movie_link on movie_link(linked_movie_id);

create index link_type_id_movie_link on movie_link(link_type_id);

create index movie_id_aka_title on aka_title(movie_id);

create index movie_id_cast_info on cast_info(movie_id);

create index movie_id_complete_cast on complete_cast(movie_id);

create index movie_id_movie_companies on movie_companies(movie_id);

create index movie_id_movie_info_idx on movie_info_idx(movie_id);

create index movie_id_movie_keyword on movie_keyword(movie_id);

create index movie_id_movie_link on movie_link(movie_id);

create index movie_id_movie_info on movie_info(movie_id);

create index person_id_aka_name on aka_name(person_id);

create index person_id_cast_info on cast_info(person_id);

create index person_id_person_info on person_info(person_id);

create index person_role_id_cast_info on cast_info(person_role_id);

create index role_id_cast_info on cast_info(role_id);
```

# TPCH

## get data
```sh
git clone https://github.com/electrum/tpch-dbgen.git
cd tpch-dbgen/
make
./dbgen -s 1 -f   
ls *.tbl

```

## Create Table
```sql
CREATE TABLE NATION  ( N_NATIONKEY  INTEGER NOT NULL,
                            N_NAME       CHAR(25) NOT NULL,
                            N_REGIONKEY  INTEGER NOT NULL,
                            N_COMMENT    VARCHAR(152));

CREATE TABLE REGION  ( R_REGIONKEY  INTEGER NOT NULL,
                            R_NAME       CHAR(25) NOT NULL,
                            R_COMMENT    VARCHAR(152));

CREATE TABLE PART  ( P_PARTKEY     INTEGER NOT NULL,
                          P_NAME        VARCHAR(55) NOT NULL,
                          P_MFGR        CHAR(25) NOT NULL,
                          P_BRAND       CHAR(10) NOT NULL,
                          P_TYPE        VARCHAR(25) NOT NULL,
                          P_SIZE        INTEGER NOT NULL,
                          P_CONTAINER   CHAR(10) NOT NULL,
                          P_RETAILPRICE DECIMAL(15,2) NOT NULL,
                          P_COMMENT     VARCHAR(23) NOT NULL );

CREATE TABLE SUPPLIER ( S_SUPPKEY     INTEGER NOT NULL,
                             S_NAME        CHAR(25) NOT NULL,
                             S_ADDRESS     VARCHAR(40) NOT NULL,
                             S_NATIONKEY   INTEGER NOT NULL,
                             S_PHONE       CHAR(15) NOT NULL,
                             S_ACCTBAL     DECIMAL(15,2) NOT NULL,
                             S_COMMENT     VARCHAR(101) NOT NULL);

CREATE TABLE PARTSUPP ( PS_PARTKEY     INTEGER NOT NULL,
                             PS_SUPPKEY     INTEGER NOT NULL,
                             PS_AVAILQTY    INTEGER NOT NULL,
                             PS_SUPPLYCOST  DECIMAL(15,2)  NOT NULL,
                             PS_COMMENT     VARCHAR(199) NOT NULL );

CREATE TABLE CUSTOMER ( C_CUSTKEY     INTEGER NOT NULL,
                             C_NAME        VARCHAR(25) NOT NULL,
                             C_ADDRESS     VARCHAR(40) NOT NULL,
                             C_NATIONKEY   INTEGER NOT NULL,
                             C_PHONE       CHAR(15) NOT NULL,
                             C_ACCTBAL     DECIMAL(15,2)   NOT NULL,
                             C_MKTSEGMENT  CHAR(10) NOT NULL,
                             C_COMMENT     VARCHAR(117) NOT NULL);

CREATE TABLE ORDERS  ( O_ORDERKEY       INTEGER NOT NULL,
                           O_CUSTKEY        INTEGER NOT NULL,
                           O_ORDERSTATUS    CHAR(1) NOT NULL,
                           O_TOTALPRICE     DECIMAL(15,2) NOT NULL,
                           O_ORDERDATE      DATE NOT NULL,
                           O_ORDERPRIORITY  CHAR(15) NOT NULL,  
                           O_CLERK          CHAR(15) NOT NULL, 
                           O_SHIPPRIORITY   INTEGER NOT NULL,
                           O_COMMENT        VARCHAR(79) NOT NULL);

CREATE TABLE LINEITEM ( L_ORDERKEY    INTEGER NOT NULL,
                             L_PARTKEY     INTEGER NOT NULL,
                             L_SUPPKEY     INTEGER NOT NULL,
                             L_LINENUMBER  INTEGER NOT NULL,
                             L_QUANTITY    DECIMAL(15,2) NOT NULL,
                             L_EXTENDEDPRICE  DECIMAL(15,2) NOT NULL,
                             L_DISCOUNT    DECIMAL(15,2) NOT NULL,
                             L_TAX         DECIMAL(15,2) NOT NULL,
                             L_RETURNFLAG  CHAR(1) NOT NULL,
                             L_LINESTATUS  CHAR(1) NOT NULL,
                             L_SHIPDATE    DATE NOT NULL,
                             L_COMMITDATE  DATE NOT NULL,
                             L_RECEIPTDATE DATE NOT NULL,
                             L_SHIPINSTRUCT CHAR(25) NOT NULL,
                             L_SHIPMODE     CHAR(10) NOT NULL,
                             L_COMMENT      VARCHAR(44) NOT NULL);

create extension pg_hint_plan;
create extension pageinspect;
su postgres


for i in `ls *.tbl`; do
    echo $i;
    sed -i 's/|$//' *.tbl;
    name=`echo $i| cut -d'.' -f1`;
    psql -h 127.0.0.1 -p 5432 -d tpch -c "COPY $name FROM '`pwd`/$i' DELIMITER '|' ENCODING 'LATIN1';";
done


\Copy region FROM '/home/usr/tpch-dbgen/region.tbl' WITH DELIMITER AS '|' ENCODING 'LATIN1';
\Copy nation FROM '/home/usr/tpch-dbgen/nation.tbl' WITH DELIMITER AS '|' ENCODING 'LATIN1';
\Copy part FROM '/home/usr/tpch-dbgen/part.tbl' WITH DELIMITER AS '|' ENCODING 'LATIN1';
\Copy supplier FROM '/home/usr/tpch-dbgen/supplier.tbl' WITH DELIMITER AS '|' ENCODING 'LATIN1';
\Copy customer FROM '/home/usr/tpch-dbgen/customer.tbl' WITH DELIMITER AS '|' ENCODING 'LATIN1';
\Copy lineitem FROM '/home/usr/tpch-dbgen/lineitem.tbl' WITH DELIMITER AS '|' ENCODING 'LATIN1';
\Copy partsupp FROM '/home/usr/tpch-dbgen/partsupp.tbl' WITH DELIMITER AS '|' ENCODING 'LATIN1';
\Copy orders FROM '/home/usr/tpch-dbgen/orders.tbl' WITH DELIMITER AS '|' ENCODING 'LATIN1';
```

## constraint
```sql
alter table PART add primary key(P_PARTKEY);

alter table SUPPLIER add primary key(S_SUPPKEY);

alter table PARTSUPP add primary key(PS_PARTKEY, PS_SUPPKEY);

alter table CUSTOMER add primary key(C_CUSTKEY);

alter table ORDERS add primary key(O_ORDERKEY);

alter table LINEITEM add primary key(L_ORDERKEY, L_LINENUMBER);

alter table NATION add primary key(N_NATIONKEY);

alter table REGION add primary key(R_REGIONKEY);

alter table SUPPLIER add CONSTRAINT f1 foreign key (S_NATIONKEY) references NATION(N_NATIONKEY);

alter table PARTSUPP add CONSTRAINT f2 foreign key (PS_PARTKEY) references PART(P_PARTKEY);

alter table PARTSUPP add CONSTRAINT f3 foreign key (PS_SUPPKEY) references SUPPLIER(S_SUPPKEY);

alter table CUSTOMER add CONSTRAINT f4 foreign key (C_NATIONKEY) references NATION(N_NATIONKEY);

alter table ORDERS add CONSTRAINT f5 foreign key (O_CUSTKEY) references CUSTOMER(C_CUSTKEY);

alter table LINEITEM add CONSTRAINT f6 foreign key (L_ORDERKEY) references ORDERS(O_ORDERKEY);

alter table LINEITEM add CONSTRAINT f7 foreign key (L_PARTKEY) references PART(P_PARTKEY);

alter table LINEITEM add CONSTRAINT f8 foreign key (L_SUPPKEY) references SUPPLIER(S_SUPPKEY);

alter table LINEITEM add CONSTRAINT f9 foreign key (L_PARTKEY, L_SUPPKEY) references PARTSUPP(PS_PARTKEY, PS_SUPPKEY);

alter table NATION add CONSTRAINT f10 foreign key (N_REGIONKEY) references REGION(R_REGIONKEY);
```

## Create Index

```sql
CREATE INDEX IDX_SUPPLIER_NATION_KEY ON SUPPLIER (S_NATIONKEY);

CREATE INDEX IDX_PARTSUPP_PARTKEY ON PARTSUPP (PS_PARTKEY);
CREATE INDEX IDX_PARTSUPP_SUPPKEY ON PARTSUPP (PS_SUPPKEY);

CREATE INDEX IDX_CUSTOMER_NATIONKEY ON CUSTOMER (C_NATIONKEY);

CREATE INDEX IDX_ORDERS_CUSTKEY ON ORDERS (O_CUSTKEY);

CREATE INDEX IDX_LINEITEM_ORDERKEY ON LINEITEM (L_ORDERKEY);
CREATE INDEX IDX_LINEITEM_PART_SUPP ON LINEITEM (L_PARTKEY,L_SUPPKEY);

CREATE INDEX IDX_NATION_REGIONKEY ON NATION (N_REGIONKEY);


-- aditional indexes

CREATE INDEX IDX_LINEITEM_SHIPDATE ON LINEITEM (L_SHIPDATE, L_DISCOUNT, L_QUANTITY);

CREATE INDEX IDX_ORDERS_ORDERDATE ON ORDERS (O_ORDERDATE);
```

# TPCDS

如果在Linux下运行，需要安装以下依赖

```sh
git clone <https://github.com/gregrahn/tpcds-kit.git>
cd tpcds-kit/tools
make OS=LINUX

create data tpcds
psql tpcds -f tpcds.sql

./dsdgen  --help

mkdir data
./dsdgen  -scale 1 -dir ./data/

cd data
```

如果是MAC OS下，则可以先从linux系统上，生成数据和查询，然后拷贝到本地文件夹上

```sh
export PGPASSWORD="postgres"
#Import data
for i in `ls *.dat`; do
  table=${i/.dat/}
  echo "Loading $table..."
  sed 's/|$//' $i > /tmp/$i
  psql -h 127.0.0.1 tpcds -q -c "TRUNCATE $table"
  psql -h 127.0.0.1 tpcds -c "\\copy $table FROM '/tmp/$i' CSV DELIMITER '|'"
done
```

```sh
# Generate Queries
#!/bin/sh
for i in `seq 1 99`
do
./dsqgen  -DIRECTORY ../query_templates/ -TEMPLATE "query${i}.tpl" -DIALECT netezza -FILTER Y > ../sql/query${i}.sql
done
```

## Create Table
```sql
create table dbgen_version
(
    dv_version                varchar(16)                   ,
    dv_create_date            date                          ,
    dv_create_time            time                          ,
    dv_cmdline_args           varchar(200)                  
);

create table customer_address
(
    ca_address_sk             integer               not null,
    ca_address_id             char(16)              not null,
    ca_street_number          char(10)                      ,
    ca_street_name            varchar(60)                   ,
    ca_street_type            char(15)                      ,
    ca_suite_number           char(10)                      ,
    ca_city                   varchar(60)                   ,
    ca_county                 varchar(30)                   ,
    ca_state                  char(2)                       ,
    ca_zip                    char(10)                      ,
    ca_country                varchar(20)                   ,
    ca_gmt_offset             decimal(5,2)                  ,
    ca_location_type          char(20)                      ,
    primary key (ca_address_sk)
);

create table customer_demographics
(
    cd_demo_sk                integer               not null,
    cd_gender                 char(1)                       ,
    cd_marital_status         char(1)                       ,
    cd_education_status       char(20)                      ,
    cd_purchase_estimate      integer                       ,
    cd_credit_rating          char(10)                      ,
    cd_dep_count              integer                       ,
    cd_dep_employed_count     integer                       ,
    cd_dep_college_count      integer                       ,
    primary key (cd_demo_sk)
);

create table date_dim
(
    d_date_sk                 integer               not null,
    d_date_id                 char(16)              not null,
    d_date                    date                          ,
    d_month_seq               integer                       ,
    d_week_seq                integer                       ,
    d_quarter_seq             integer                       ,
    d_year                    integer                       ,
    d_dow                     integer                       ,
    d_moy                     integer                       ,
    d_dom                     integer                       ,
    d_qoy                     integer                       ,
    d_fy_year                 integer                       ,
    d_fy_quarter_seq          integer                       ,
    d_fy_week_seq             integer                       ,
    d_day_name                char(9)                       ,
    d_quarter_name            char(6)                       ,
    d_holiday                 char(1)                       ,
    d_weekend                 char(1)                       ,
    d_following_holiday       char(1)                       ,
    d_first_dom               integer                       ,
    d_last_dom                integer                       ,
    d_same_day_ly             integer                       ,
    d_same_day_lq             integer                       ,
    d_current_day             char(1)                       ,
    d_current_week            char(1)                       ,
    d_current_month           char(1)                       ,
    d_current_quarter         char(1)                       ,
    d_current_year            char(1)                       ,
    primary key (d_date_sk)
);

create table warehouse
(
    w_warehouse_sk            integer               not null,
    w_warehouse_id            char(16)              not null,
    w_warehouse_name          varchar(20)                   ,
    w_warehouse_sq_ft         integer                       ,
    w_street_number           char(10)                      ,
    w_street_name             varchar(60)                   ,
    w_street_type             char(15)                      ,
    w_suite_number            char(10)                      ,
    w_city                    varchar(60)                   ,
    w_county                  varchar(30)                   ,
    w_state                   char(2)                       ,
    w_zip                     char(10)                      ,
    w_country                 varchar(20)                   ,
    w_gmt_offset              decimal(5,2)                  ,
    primary key (w_warehouse_sk)
);

create table ship_mode
(
    sm_ship_mode_sk           integer               not null,
    sm_ship_mode_id           char(16)              not null,
    sm_type                   char(30)                      ,
    sm_code                   char(10)                      ,
    sm_carrier                char(20)                      ,
    sm_contract               char(20)                      ,
    primary key (sm_ship_mode_sk)
);

create table time_dim
(
    t_time_sk                 integer               not null,
    t_time_id                 char(16)              not null,
    t_time                    integer                       ,
    t_hour                    integer                       ,
    t_minute                  integer                       ,
    t_second                  integer                       ,
    t_am_pm                   char(2)                       ,
    t_shift                   char(20)                      ,
    t_sub_shift               char(20)                      ,
    t_meal_time               char(20)                      ,
    primary key (t_time_sk)
);

create table reason
(
    r_reason_sk               integer               not null,
    r_reason_id               char(16)              not null,
    r_reason_desc             char(100)                     ,
    primary key (r_reason_sk)
);

create table income_band
(
    ib_income_band_sk         integer               not null,
    ib_lower_bound            integer                       ,
    ib_upper_bound            integer                       ,
    primary key (ib_income_band_sk)
);

create table item
(
    i_item_sk                 integer               not null,
    i_item_id                 char(16)              not null,
    i_rec_start_date          date                          ,
    i_rec_end_date            date                          ,
    i_item_desc               varchar(200)                  ,
    i_current_price           decimal(7,2)                  ,
    i_wholesale_cost          decimal(7,2)                  ,
    i_brand_id                integer                       ,
    i_brand                   char(50)                      ,
    i_class_id                integer                       ,
    i_class                   char(50)                      ,
    i_category_id             integer                       ,
    i_category                char(50)                      ,
    i_manufact_id             integer                       ,
    i_manufact                char(50)                      ,
    i_size                    char(20)                      ,
    i_formulation             char(20)                      ,
    i_color                   char(20)                      ,
    i_units                   char(10)                      ,
    i_container               char(10)                      ,
    i_manager_id              integer                       ,
    i_product_name            char(50)                      ,
    primary key (i_item_sk)
);

create table store
(
    s_store_sk                integer               not null,
    s_store_id                char(16)              not null,
    s_rec_start_date          date                          ,
    s_rec_end_date            date                          ,
    s_closed_date_sk          integer                       ,
    s_store_name              varchar(50)                   ,
    s_number_employees        integer                       ,
    s_floor_space             integer                       ,
    s_hours                   char(20)                      ,
    s_manager                 varchar(40)                   ,
    s_market_id               integer                       ,
    s_geography_class         varchar(100)                  ,
    s_market_desc             varchar(100)                  ,
    s_market_manager          varchar(40)                   ,
    s_division_id             integer                       ,
    s_division_name           varchar(50)                   ,
    s_company_id              integer                       ,
    s_company_name            varchar(50)                   ,
    s_street_number           varchar(10)                   ,
    s_street_name             varchar(60)                   ,
    s_street_type             char(15)                      ,
    s_suite_number            char(10)                      ,
    s_city                    varchar(60)                   ,
    s_county                  varchar(30)                   ,
    s_state                   char(2)                       ,
    s_zip                     char(10)                      ,
    s_country                 varchar(20)                   ,
    s_gmt_offset              decimal(5,2)                  ,
    s_tax_precentage          decimal(5,2)                  ,
    primary key (s_store_sk)
);

create table call_center
(
    cc_call_center_sk         integer               not null,
    cc_call_center_id         char(16)              not null,
    cc_rec_start_date         date                          ,
    cc_rec_end_date           date                          ,
    cc_closed_date_sk         integer                       ,
    cc_open_date_sk           integer                       ,
    cc_name                   varchar(50)                   ,
    cc_class                  varchar(50)                   ,
    cc_employees              integer                       ,
    cc_sq_ft                  integer                       ,
    cc_hours                  char(20)                      ,
    cc_manager                varchar(40)                   ,
    cc_mkt_id                 integer                       ,
    cc_mkt_class              char(50)                      ,
    cc_mkt_desc               varchar(100)                  ,
    cc_market_manager         varchar(40)                   ,
    cc_division               integer                       ,
    cc_division_name          varchar(50)                   ,
    cc_company                integer                       ,
    cc_company_name           char(50)                      ,
    cc_street_number          char(10)                      ,
    cc_street_name            varchar(60)                   ,
    cc_street_type            char(15)                      ,
    cc_suite_number           char(10)                      ,
    cc_city                   varchar(60)                   ,
    cc_county                 varchar(30)                   ,
    cc_state                  char(2)                       ,
    cc_zip                    char(10)                      ,
    cc_country                varchar(20)                   ,
    cc_gmt_offset             decimal(5,2)                  ,
    cc_tax_percentage         decimal(5,2)                  ,
    primary key (cc_call_center_sk)
);

create table customer
(
    c_customer_sk             integer               not null,
    c_customer_id             char(16)              not null,
    c_current_cdemo_sk        integer                       ,
    c_current_hdemo_sk        integer                       ,
    c_current_addr_sk         integer                       ,
    c_first_shipto_date_sk    integer                       ,
    c_first_sales_date_sk     integer                       ,
    c_salutation              char(10)                      ,
    c_first_name              char(20)                      ,
    c_last_name               char(30)                      ,
    c_preferred_cust_flag     char(1)                       ,
    c_birth_day               integer                       ,
    c_birth_month             integer                       ,
    c_birth_year              integer                       ,
    c_birth_country           varchar(20)                   ,
    c_login                   char(13)                      ,
    c_email_address           char(50)                      ,
    c_last_review_date        char(10)                      ,
    primary key (c_customer_sk)
);

create table web_site
(
    web_site_sk               integer               not null,
    web_site_id               char(16)              not null,
    web_rec_start_date        date                          ,
    web_rec_end_date          date                          ,
    web_name                  varchar(50)                   ,
    web_open_date_sk          integer                       ,
    web_close_date_sk         integer                       ,
    web_class                 varchar(50)                   ,
    web_manager               varchar(40)                   ,
    web_mkt_id                integer                       ,
    web_mkt_class             varchar(50)                   ,
    web_mkt_desc              varchar(100)                  ,
    web_market_manager        varchar(40)                   ,
    web_company_id            integer                       ,
    web_company_name          char(50)                      ,
    web_street_number         char(10)                      ,
    web_street_name           varchar(60)                   ,
    web_street_type           char(15)                      ,
    web_suite_number          char(10)                      ,
    web_city                  varchar(60)                   ,
    web_county                varchar(30)                   ,
    web_state                 char(2)                       ,
    web_zip                   char(10)                      ,
    web_country               varchar(20)                   ,
    web_gmt_offset            decimal(5,2)                  ,
    web_tax_percentage        decimal(5,2)                  ,
    primary key (web_site_sk)
);

create table store_returns
(
    sr_returned_date_sk       integer                       ,
    sr_return_time_sk         integer                       ,
    sr_item_sk                integer               not null,
    sr_customer_sk            integer                       ,
    sr_cdemo_sk               integer                       ,
    sr_hdemo_sk               integer                       ,
    sr_addr_sk                integer                       ,
    sr_store_sk               integer                       ,
    sr_reason_sk              integer                       ,
    sr_ticket_number          integer               not null,
    sr_return_quantity        integer                       ,
    sr_return_amt             decimal(7,2)                  ,
    sr_return_tax             decimal(7,2)                  ,
    sr_return_amt_inc_tax     decimal(7,2)                  ,
    sr_fee                    decimal(7,2)                  ,
    sr_return_ship_cost       decimal(7,2)                  ,
    sr_refunded_cash          decimal(7,2)                  ,
    sr_reversed_charge        decimal(7,2)                  ,
    sr_store_credit           decimal(7,2)                  ,
    sr_net_loss               decimal(7,2)                  ,
    primary key (sr_item_sk, sr_ticket_number)
);

create table household_demographics
(
    hd_demo_sk                integer               not null,
    hd_income_band_sk         integer                       ,
    hd_buy_potential          char(15)                      ,
    hd_dep_count              integer                       ,
    hd_vehicle_count          integer                       ,
    primary key (hd_demo_sk)
);

create table web_page
(
    wp_web_page_sk            integer               not null,
    wp_web_page_id            char(16)              not null,
    wp_rec_start_date         date                          ,
    wp_rec_end_date           date                          ,
    wp_creation_date_sk       integer                       ,
    wp_access_date_sk         integer                       ,
    wp_autogen_flag           char(1)                       ,
    wp_customer_sk            integer                       ,
    wp_url                    varchar(100)                  ,
    wp_type                   char(50)                      ,
    wp_char_count             integer                       ,
    wp_link_count             integer                       ,
    wp_image_count            integer                       ,
    wp_max_ad_count           integer                       ,
    primary key (wp_web_page_sk)
);

create table promotion
(
    p_promo_sk                integer               not null,
    p_promo_id                char(16)              not null,
    p_start_date_sk           integer                       ,
    p_end_date_sk             integer                       ,
    p_item_sk                 integer                       ,
    p_cost                    decimal(15,2)                 ,
    p_response_target         integer                       ,
    p_promo_name              char(50)                      ,
    p_channel_dmail           char(1)                       ,
    p_channel_email           char(1)                       ,
    p_channel_catalog         char(1)                       ,
    p_channel_tv              char(1)                       ,
    p_channel_radio           char(1)                       ,
    p_channel_press           char(1)                       ,
    p_channel_event           char(1)                       ,
    p_channel_demo            char(1)                       ,
    p_channel_details         varchar(100)                  ,
    p_purpose                 char(15)                      ,
    p_discount_active         char(1)                       ,
    primary key (p_promo_sk)
);

create table catalog_page
(
    cp_catalog_page_sk        integer               not null,
    cp_catalog_page_id        char(16)              not null,
    cp_start_date_sk          integer                       ,
    cp_end_date_sk            integer                       ,
    cp_department             varchar(50)                   ,
    cp_catalog_number         integer                       ,
    cp_catalog_page_number    integer                       ,
    cp_description            varchar(100)                  ,
    cp_type                   varchar(100)                  ,
    primary key (cp_catalog_page_sk)
);

create table inventory
(
    inv_date_sk               integer               not null,
    inv_item_sk               integer               not null,
    inv_warehouse_sk          integer               not null,
    inv_quantity_on_hand      integer                       ,
    primary key (inv_date_sk, inv_item_sk, inv_warehouse_sk)
);

create table catalog_returns
(
    cr_returned_date_sk       integer                       ,
    cr_returned_time_sk       integer                       ,
    cr_item_sk                integer               not null,
    cr_refunded_customer_sk   integer                       ,
    cr_refunded_cdemo_sk      integer                       ,
    cr_refunded_hdemo_sk      integer                       ,
    cr_refunded_addr_sk       integer                       ,
    cr_returning_customer_sk  integer                       ,
    cr_returning_cdemo_sk     integer                       ,
    cr_returning_hdemo_sk     integer                       ,
    cr_returning_addr_sk      integer                       ,
    cr_call_center_sk         integer                       ,
    cr_catalog_page_sk        integer                       ,
    cr_ship_mode_sk           integer                       ,
    cr_warehouse_sk           integer                       ,
    cr_reason_sk              integer                       ,
    cr_order_number           integer               not null,
    cr_return_quantity        integer                       ,
    cr_return_amount          decimal(7,2)                  ,
    cr_return_tax             decimal(7,2)                  ,
    cr_return_amt_inc_tax     decimal(7,2)                  ,
    cr_fee                    decimal(7,2)                  ,
    cr_return_ship_cost       decimal(7,2)                  ,
    cr_refunded_cash          decimal(7,2)                  ,
    cr_reversed_charge        decimal(7,2)                  ,
    cr_store_credit           decimal(7,2)                  ,
    cr_net_loss               decimal(7,2)                  ,
    primary key (cr_item_sk, cr_order_number)
);

create table web_returns
(
    wr_returned_date_sk       integer                       ,
    wr_returned_time_sk       integer                       ,
    wr_item_sk                integer               not null,
    wr_refunded_customer_sk   integer                       ,
    wr_refunded_cdemo_sk      integer                       ,
    wr_refunded_hdemo_sk      integer                       ,
    wr_refunded_addr_sk       integer                       ,
    wr_returning_customer_sk  integer                       ,
    wr_returning_cdemo_sk     integer                       ,
    wr_returning_hdemo_sk     integer                       ,
    wr_returning_addr_sk      integer                       ,
    wr_web_page_sk            integer                       ,
    wr_reason_sk              integer                       ,
    wr_order_number           integer               not null,
    wr_return_quantity        integer                       ,
    wr_return_amt             decimal(7,2)                  ,
    wr_return_tax             decimal(7,2)                  ,
    wr_return_amt_inc_tax     decimal(7,2)                  ,
    wr_fee                    decimal(7,2)                  ,
    wr_return_ship_cost       decimal(7,2)                  ,
    wr_refunded_cash          decimal(7,2)                  ,
    wr_reversed_charge        decimal(7,2)                  ,
    wr_account_credit         decimal(7,2)                  ,
    wr_net_loss               decimal(7,2)                  ,
    primary key (wr_item_sk, wr_order_number)
);

create table web_sales
(
    ws_sold_date_sk           integer                       ,
    ws_sold_time_sk           integer                       ,
    ws_ship_date_sk           integer                       ,
    ws_item_sk                integer               not null,
    ws_bill_customer_sk       integer                       ,
    ws_bill_cdemo_sk          integer                       ,
    ws_bill_hdemo_sk          integer                       ,
    ws_bill_addr_sk           integer                       ,
    ws_ship_customer_sk       integer                       ,
    ws_ship_cdemo_sk          integer                       ,
    ws_ship_hdemo_sk          integer                       ,
    ws_ship_addr_sk           integer                       ,
    ws_web_page_sk            integer                       ,
    ws_web_site_sk            integer                       ,
    ws_ship_mode_sk           integer                       ,
    ws_warehouse_sk           integer                       ,
    ws_promo_sk               integer                       ,
    ws_order_number           integer               not null,
    ws_quantity               integer                       ,
    ws_wholesale_cost         decimal(7,2)                  ,
    ws_list_price             decimal(7,2)                  ,
    ws_sales_price            decimal(7,2)                  ,
    ws_ext_discount_amt       decimal(7,2)                  ,
    ws_ext_sales_price        decimal(7,2)                  ,
    ws_ext_wholesale_cost     decimal(7,2)                  ,
    ws_ext_list_price         decimal(7,2)                  ,
    ws_ext_tax                decimal(7,2)                  ,
    ws_coupon_amt             decimal(7,2)                  ,
    ws_ext_ship_cost          decimal(7,2)                  ,
    ws_net_paid               decimal(7,2)                  ,
    ws_net_paid_inc_tax       decimal(7,2)                  ,
    ws_net_paid_inc_ship      decimal(7,2)                  ,
    ws_net_paid_inc_ship_tax  decimal(7,2)                  ,
    ws_net_profit             decimal(7,2)                  ,
    primary key (ws_item_sk, ws_order_number)
);

create table catalog_sales
(
    cs_sold_date_sk           integer                       ,
    cs_sold_time_sk           integer                       ,
    cs_ship_date_sk           integer                       ,
    cs_bill_customer_sk       integer                       ,
    cs_bill_cdemo_sk          integer                       ,
    cs_bill_hdemo_sk          integer                       ,
    cs_bill_addr_sk           integer                       ,
    cs_ship_customer_sk       integer                       ,
    cs_ship_cdemo_sk          integer                       ,
    cs_ship_hdemo_sk          integer                       ,
    cs_ship_addr_sk           integer                       ,
    cs_call_center_sk         integer                       ,
    cs_catalog_page_sk        integer                       ,
    cs_ship_mode_sk           integer                       ,
    cs_warehouse_sk           integer                       ,
    cs_item_sk                integer               not null,
    cs_promo_sk               integer                       ,
    cs_order_number           integer               not null,
    cs_quantity               integer                       ,
    cs_wholesale_cost         decimal(7,2)                  ,
    cs_list_price             decimal(7,2)                  ,
    cs_sales_price            decimal(7,2)                  ,
    cs_ext_discount_amt       decimal(7,2)                  ,
    cs_ext_sales_price        decimal(7,2)                  ,
    cs_ext_wholesale_cost     decimal(7,2)                  ,
    cs_ext_list_price         decimal(7,2)                  ,
    cs_ext_tax                decimal(7,2)                  ,
    cs_coupon_amt             decimal(7,2)                  ,
    cs_ext_ship_cost          decimal(7,2)                  ,
    cs_net_paid               decimal(7,2)                  ,
    cs_net_paid_inc_tax       decimal(7,2)                  ,
    cs_net_paid_inc_ship      decimal(7,2)                  ,
    cs_net_paid_inc_ship_tax  decimal(7,2)                  ,
    cs_net_profit             decimal(7,2)                  ,
    primary key (cs_item_sk, cs_order_number)
);

create table store_sales
(
    ss_sold_date_sk           integer                       ,
    ss_sold_time_sk           integer                       ,
    ss_item_sk                integer               not null,
    ss_customer_sk            integer                       ,
    ss_cdemo_sk               integer                       ,
    ss_hdemo_sk               integer                       ,
    ss_addr_sk                integer                       ,
    ss_store_sk               integer                       ,
    ss_promo_sk               integer                       ,
    ss_ticket_number          integer               not null,
    ss_quantity               integer                       ,
    ss_wholesale_cost         decimal(7,2)                  ,
    ss_list_price             decimal(7,2)                  ,
    ss_sales_price            decimal(7,2)                  ,
    ss_ext_discount_amt       decimal(7,2)                  ,
    ss_ext_sales_price        decimal(7,2)                  ,
    ss_ext_wholesale_cost     decimal(7,2)                  ,
    ss_ext_list_price         decimal(7,2)                  ,
    ss_ext_tax                decimal(7,2)                  ,
    ss_coupon_amt             decimal(7,2)                  ,
    ss_net_paid               decimal(7,2)                  ,
    ss_net_paid_inc_tax       decimal(7,2)                  ,
    ss_net_profit             decimal(7,2)                  ,
    primary key (ss_item_sk, ss_ticket_number)
);
```


## Create Index

```sql
CREATE INDEX c_customer_sk_idx ON customer(c_customer_sk);
CREATE INDEX d_date_sk_idx ON date_dim(d_date_sk);
CREATE INDEX d_date_idx ON date_dim(d_date);
CREATE INDEX d_month_seq_idx ON date_dim(d_month_seq);
CREATE INDEX d_year_idx ON date_dim(d_year);
CREATE INDEX i_item_sk_idx ON item(i_item_sk);
CREATE INDEX s_state_idx ON store(s_state);
CREATE INDEX s_store_sk_idx ON store(s_store_sk);
CREATE INDEX sr_returned_date_sk_idx ON store_returns(sr_returned_date_sk);
CREATE INDEX ss_sold_date_sk_idx ON store_sales(ss_sold_date_sk);

```