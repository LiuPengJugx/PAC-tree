from pyspark.sql import SparkSession
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import os
import time
import pandas as pd
import sys
sys.path.append('/home/liupengju/pycharmProjects/dasfa/')
from model.hp_partitioner import Partitioner
from tool.Conf import col_inf,db_configuration
from model.par_tree import PartitionTree
from tool.utils import *
from tool.utils import *
import argparse
from model.predictor import encode_real_queries,gen_raw_queries,recover_query_from_norm


parser = argparse.ArgumentParser()
parser.add_argument('--benchmark',type=str, help='benchmark help')
parser.add_argument('--table', type=str,help='table help')
parser.add_argument('--querytype', type=int,help='query type help')
args = parser.parse_args()

spark = SparkSession \
    .builder \
    .master("local") \
    .appName("Python Spark SQL Execution") \
    .config("spark.executor.memory", "8g") \
    .config("spark.driver.memory","8g") \
    .config("spark.memory.offHeap.enabled",True) \
    .config("spark.memory.offHeap.size","8g") \
    .getOrCreate()

os.environ['HADOOP_HOME'] = '/home/liupengju/hadoop-3.3.1'
os.environ['JAVA_HOME'] = '/home/liupengju/jdk1.8.0_181'
os.environ['ARROW_LIBHDFS_DIR'] = '/home/liupengju/hadoop-3.3.1/lib/native'

# set configurations
hdfs_private_ip = '127.0.0.1'
port=9000
hdfs_base_path = f'hdfs://{hdfs_private_ip}:{port}/par_text'

benchmark=args.benchmark
tab=args.table
querytype=args.querytype

print_lower_bound=False
strategy_tree_dict={1:'qdtree',2:'paw',3:'torn',5:'torn_r'}
vp=list(range(len(col_inf[tab]['name'])))
layout_base_path='/home/liupengju/pycharmProjects/dasfa/layouts'
block_size=math.ceil(db_configuration['block_size']/sum(col_inf[tab]['length']))
print('block size====',block_size)
# dataset,encode_dicts=load_data_dict(benchmark,tab)
dataset=load_data(benchmark,tab)
encode_dicts=load_dict(benchmark,tab)
boundary=col_parser(dataset,encode_dicts)
domains=normalize_continuation_boundary(boundary)
print(domains)
fs = pa.hdfs.connect(host=hdfs_private_ip, port=port, user='liupengju')

def find_overlap_parquets(query, partition_index):
    '''
    find out all the overlap partition ids
    '''
    query_lower = [qr[0] for qr in query]
    query_upper = [qr[1] for qr in query]
    query_border = tuple(query_lower + query_upper)
    overlap_pids = list(partition_index.intersection(query_border))

    return overlap_pids


def transform_query_to_sql(hdfs_path, query, num_dims, column_names, querytype=0, pids=None):
    sql = ''
    where_clause=''
    filter_columns=[]
    # print(f'query:{query}   -------- domains:{domains}')
    for dim in range(num_dims):
        if query[dim] is None: continue
        if query[dim][0]>domains[dim][0]:
            where_clause += column_names[dim] + '>=' + str(query[dim][0]) + ' and '
        if query[dim][1]<domains[dim][1]:
            where_clause += column_names[dim] + '<=' + str(query[dim][1]) + ' and '
        if query[dim][0]>domains[dim][0] or query[dim][1]<domains[dim][1]: filter_columns.append(column_names[dim])
    where_clause = where_clause[0:-4]
    
    if pids is not None and len(pids) != 0:
        pids_str = str(set(pids)).replace(" ", "")  # '{1,2,3}'
        pq_hdfs_path = hdfs_path + '/partition_' + pids_str + ".parquet"

    if querytype == 0:
        sql = "SELECT * FROM parquet.`" + pq_hdfs_path + "`WHERE " + where_clause
    elif querytype == 1:
        sql = "SELECT COUNT(*) FROM parquet.`" + pq_hdfs_path + "`WHERE " + where_clause
    elif querytype == 2:
        sql = "SELECT variance(_c0) FROM parquet.`" + pq_hdfs_path + "`WHERE " + where_clause
    elif querytype == 3:
        sql = ["SELECT * FROM parquet.`" + pq_hdfs_path + "`"]
    elif querytype == 4:
        sql = "SELECT _c0 FROM parquet.`" + pq_hdfs_path + "`"
    elif querytype==5:
        sql=[]
        for pid in pids:
            sql.append(f"SELECT * FROM parquet.`{hdfs_path + '/partition_' + str(pid) + '.parquet'}`") 

#     print(sql,where_clause,filter_columns)
    return sql,where_clause,filter_columns


def query_with_parquets(hdfs_path,query, num_dims, column_names, querytype=0, partition_tree=None,
                        print_execution_time=False):
    start_time = time.time()
    sql = None
    where_clause=None
    repeated_times=1
    filter_data_size=[]
    pids = partition_tree.query_single(query[0])
    vaild=True
    if partition_tree == None:
        sql,_,_ = transform_query_to_sql(hdfs_path, query[1], num_dims, column_names, querytype)
    else:
#       print("pids:", pids)
        if len(pids)>0:
            sql,where_clause,filter_columns = transform_query_to_sql(hdfs_path, query[1], num_dims, column_names, querytype, pids)
    # print("generated sql:", sql)
    end_time_1 = time.time()
    exe_time_list=[]
    query_execution_time=0
    if isinstance (sql,list):
        for _ in range(repeated_times):
            start_time_1=time.time()
            if print_lower_bound:
                query_result=[]
                for sql_item in sql:
                    query_result += spark.sql(sql_item).select(filter_columns).where(where_clause).collect()  #distinct()
                    filter_data_size.append(len(query_result))
                if len(query_result)<=0: vaild=False
                else:
                    pdf = pd.DataFrame(query_result, columns=filter_columns)
                    b_size=block_size*2
                    lb_hdfs_path=f"{hdfs_base_path}/{suffix}/{benchmark}/{tab}/LB-Cost/merged"
                    b_num=math.ceil(pdf.shape[0]/b_size)
                    pids=range(b_num)
                    def update_temp_files(mode='mkdir'):
                        for b_counter in range(b_num):
                            lb_partition_path = lb_hdfs_path + f"/partition_lb_{b_counter}.parquet"
                            if mode=='mkdir':
                                with fs.open(lb_partition_path,'wb') as f:
                                    start_idx=b_size*b_counter
                                    end_idx=b_size*(b_counter+1) if b_counter!=b_num-1 else -1
                                    adf = pa.Table.from_pandas(pdf[start_idx:end_idx])
                                    pq.write_table(adf, f,write_statistics=False,use_dictionary=False,compression='none')
                            
                        if mode=='rm':
                            fs.delete(lb_hdfs_path,recursive=True)
                    
                    update_temp_files(mode='mkdir')
                    if querytype==5:
                        cul_time=0
                        for b_counter in range(b_num):
                            lb_sql=f"SELECT * FROM parquet.`{lb_hdfs_path + '/partition_lb_' + str(b_counter) + '.parquet'}`"
                            lb_start_time_1=time.time()
                            spark.sql(lb_sql).select(filter_columns).where(where_clause).collect()
                            cul_time+=time.time()-lb_start_time_1
                    else:
                        lb_sql="SELECT * FROM parquet.`" + lb_hdfs_path + '/partition_lb_' + str(set(range(b_num))).replace(" ", "") + ".parquet" + "`"
                        lb_start_time_1=time.time()
                        spark.sql(lb_sql).select(filter_columns).where(where_clause).collect()
                        cul_time=time.time()-lb_start_time_1
                    query_execution_time=cul_time
                    update_temp_files(mode='rm')
            else:
                # print(f"sql_item:{sql}\n filter_columns:{filter_columns}\n where_clause:{where_clause}")
                for sql_item in sql:
                    query_result = spark.sql(sql_item).select(filter_columns).where(where_clause).collect()  #distinct()
                    filter_data_size.append(len(query_result))
                query_execution_time=time.time()-start_time_1
    elif sql:
        filter_data_size.append(len(spark.sql(sql).collect()))
    else:
        print('sql is None!')
        vaild=False
    end_time_2 = time.time()
#     print("result size:", sum(filter_data_size))

    # 1. compute actual tuples from parquet file
    # pids = partition_tree.query_single(query)
    # pids = str(set(pids)).replace(" ", "")  # '{1,2,3}'
    # parquets_path = hdfs_path + '/partition_' + pids + ".parquet"
    # count_sql = "SELECT COUNT(*) FROM parquet.`" + parquets_path+"`"
    # actual_data_size=spark.sql(count_sql).collect()[0]['count(1)']
    actual_data_size1=sum(filter_data_size)
    
    # 2. compute actual tuples from parquet meta file
    
    data_size_list=[]
    # parquets_path=[hdfs_path + '/partition_' + str(pid) + ".parquet" for pid in pids]
    # actual_data_size2=0
    # for par_path in parquets_path:
    #     fw=fs.open(par_path,'rb')
    #     meta = pa.parquet.read_metadata(fw, memory_map=False).to_dict()
    #     actual_data_size2+=meta['num_rows']
    #     data_size_list.append(meta['num_rows'])
    #     fw.close()
    query_translation_time = end_time_1 - start_time
    # query_execution_time = end_time_2 - end_time_1
#     print('query execution time: ', query_execution_time,' parquet size:',sum(data_size_list))
    # print(f"{data_size_list}--{filter_data_size}")
    if print_execution_time:
        print('query translation time: ', query_translation_time)
        print('query execution time: ', query_execution_time)
#     return (query_translation_time, query_execution_time,query_result.toPandas().shape[0],actual_data_size2,sum(data_size_list))
    print(f"pids length: {len(pids)};  latency: {query_execution_time/repeated_times}")
    return (query_translation_time, query_execution_time/repeated_times, actual_data_size1,data_size_list,vaild)

def batch_query(queryset, benchmark,tab,partition_tree, hdfs_path,querytype=0):
    # add statistics result
    results = []

    count = 0
    col_names=col_inf[tab]['name']
    num_dims=len(col_names)

    for i in range(0, len(queryset)):
        result = query_with_parquets(hdfs_path, queryset[i], num_dims, col_names, querytype, partition_tree)
        if result[4]:
            # print('finish query#', count,' execution time:',result[1],' node list:',result[3])
            print('finish query#', count,' execution time->',result[1])
            results.append(result)
        count += 1

    result_size = 0
    actual_data_size=0
    mean_response_time=[]
    # min_response_time=[]
    accumulate_response_time=[]
    for result in results:
        # min_response_time.append(result[1][0])
        mean_response_time.append(result[1])
        accumulate_response_time.append(sum(mean_response_time)/len(mean_response_time))
    latency_df=pd.Series(accumulate_response_time).quantile([0.75, 0.90, 0.95])
#         result_size += result[2]
#         actual_data_size+=result[3]
#     avg_result_size = int(result_size // len(queryset))
#     avg_actual_data_size = int(actual_data_size // len(queryset))
    # min_response_time=sum(min_response_time)/len(min_response_time)
    return_resp=[latency_df.iloc[0],latency_df.iloc[1],latency_df.iloc[2],min(mean_response_time),sum(mean_response_time)/len(mean_response_time)]
    print(f'Query response time: 75th->{return_resp[0]:5.3f} 90th->{return_resp[1]:5.3f} 95th->{return_resp[2]:5.3f} min->{return_resp[3]:5.3f}, mean->{return_resp[4]:5.3f}')
    return return_resp

def getQueryTuple(benchmark,tab,partition_tree,suffix):
    queryset=[]
    if suffix=='numeric':    
        default_workload_path=f'/home/liupengju/pycharmProjects/dasfa/queries/{benchmark}-encode-queries/{tab}_{suffix}.csv'
    else:
        default_workload_path=f'/home/liupengju/pycharmProjects/dasfa/queries/{benchmark}-encode-queries/{tab}.csv'
    queries=Partitioner().load_encode_queryset(default_workload_path,benchmark,tab)
    for qid,q in enumerate(queries):
        print(qid)
        query_boundary=partition_tree.pt_root.approximate_bound_for_query(q,encode_dicts,True)
        queryset.append((q,query_boundary))
    return queryset

def getDynamicQueryTuple(benchmark,tab,partition_tree):
    queryset=[]
    query_common_path='/home/liupengju/pycharmProjects/dasfa/queries/similar-queries'
    _,domains,domain_ids=encode_real_queries(suffix='similar-queries',tab=tab,benchmark=benchmark)
    with open(f"{query_common_path}/{benchmark}_{tab}_test.pickle",'rb') as f:
        train_test_set=pickle.load(f)
        train_set, test_set = np.hsplit(recover_query_from_norm(train_test_set, len(domains), domains), 2)
    queries=gen_raw_queries(test_set,tab,domains,domain_ids)
#     print(encode_dicts)
    for q in queries:
        query_boundary=partition_tree.pt_root.approximate_bound_for_query(q,encode_dicts,True)
        queryset.append((q,query_boundary))
    return queryset

partition_tree=PartitionTree(vp=vp)
partition_tree.dataset=dataset
exp_result_dict={}
# static load
suffix='static'
# querytype=3 #hits/lineitem
# querytype=5

# suffix='numeric'
# querytype=3 # hits

# dynamic load
# suffix='similar'
# querytype=3 #hits/lineitem

queryset=None
for strategy in [5]: #[2,5,6]
    if strategy==6: #lb-COST 
        print_lower_bound=True  # test LB-Cost
        partition_tree.load_tree(f'{layout_base_path}/{suffix}/{benchmark}_{tab}_{strategy_tree_dict[2]}')
        hdfs_path=f"{hdfs_base_path}/{suffix}/{benchmark}/{tab}/{strategy_tree_dict[2]}/merged"
    else:
        partition_tree.load_tree(f'{layout_base_path}/{suffix}/{benchmark}_{tab}_{strategy_tree_dict[strategy]}')
        hdfs_path=f"{hdfs_base_path}/{suffix}/{benchmark}/{tab}/{strategy_tree_dict[strategy]}/merged"
    if queryset is None:
        if suffix=='similar':
            queryset=getDynamicQueryTuple(benchmark,tab,partition_tree)
        else:
            queryset=getQueryTuple(benchmark,tab,partition_tree,suffix)
        # print(queryset)
    latency=batch_query(queryset, benchmark, tab, partition_tree, hdfs_path, querytype=querytype)
    exp_result_dict[strategy]=latency
if suffix=='numeric':
    pd.DataFrame(data=exp_result_dict.values(),index=list(exp_result_dict.keys())).to_csv("/home/liupengju/pycharmProjects/dasfa/spark/spark_latency/"+benchmark+"_"+tab+"_numeric_res.csv",header=['75th','90th','95th','min','mean'])
else:
    pd.DataFrame(data=exp_result_dict.values(),index=list(exp_result_dict.keys())).to_csv("/home/liupengju/pycharmProjects/dasfa/spark/spark_latency/"+benchmark+"_"+tab+"_res.csv",header=['75th','90th','95th','min','mean'])
    
print(exp_result_dict)
