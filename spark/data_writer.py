from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
import os
import sys
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import time
import argparse
import numpy as np
import pickle

sys.path.append(os.path.dirname(os.path.abspath(__file__))+ '/..')
from model.partition_tree import PartitionTree

parser = argparse.ArgumentParser()
parser.add_argument('--benchmark',type=str, help='benchmark help')
args = parser.parse_args()


benchmark=args.benchmark

hdfs_private_ip = '127.0.0.1'
port=9000
hdfs_base_path = f'hdfs://{hdfs_private_ip}:{port}/join_layout_proj'
chunk_size = 10000
num_process=10
base_dir=os.path.dirname(os.path.abspath(__file__))

table_metadata=pickle.load(open(f'{base_dir}/../dataset/{benchmark}/metadata.pkl','rb'))


def init_spark():
    conf = SparkConf().setAll([("spark.executor.memory", "24g"),("spark.driver.memory","24g"),
                           ("spark.memory.offHeap.enabled",True),("spark.memory.offHeap.size","16g"),
                          ("spark.driver.maxResultSize", "16g")])
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    os.environ['HADOOP_HOME'] = '~/hadoop-3.3.1'
    os.environ['JAVA_HOME'] = '~/jdk1.8.0_181'
    os.environ['ARROW_LIBHDFS_DIR'] = '~/hadoop-3.3.1/lib/native'
    return sc,sqlContext

def process_chunk_row(row, partition_tree, pid_data_dict, count, k):
    count[0] += 1
    row_numpy = row.to_numpy()
    row = row_numpy.tolist()
    pids = []
    try:
        pids = partition_tree.get_pid_for_data_point(row)
    except:
        print(row)
    if isinstance(pids,list):
        for pid in pids:
            if pid in pid_data_dict:
                pid_data_dict[pid]+=[row]
            else:
                pid_data_dict[pid]=[row]


def process_chunk(chunk, k, partition_tree):
    print("enter data routing process, process chunk ", k, '..')
    pid_data_dict = {}
    count = [0]
    chunk.apply(lambda row: process_chunk_row(row, partition_tree, pid_data_dict, count, k), axis=1)
    print("exit process chunk ", k, ".")
    return pid_data_dict


def merge_epochs(parameters):
    pids, epoch_count, hdfs_path, fs, merge_process = parameters
    for pid in pids:
        parquets = []
        for epoch in range(epoch_count):
            path = hdfs_path + "/epoch_" + str(epoch) + '/partition_' + str(pid)+'.parquet'
            #print(path)
            try:
                par = pq.read_table(path)
                parquets.append(par)
            except:
                continue
        print("process", merge_process, "pid", pid, " len parquets (epochs):", len(parquets))
        if len(parquets) == 0:
            continue
        merged_parquet = pa.concat_tables(parquets)
        merge_path = hdfs_path + '/merged/partition_' + str(pid)+'.parquet'
        with fs.open(merge_path,'wb') as f:
            pq.write_table(merged_parquet, f)
    print('exit merge process', merge_process)

def merge_dict(base_dict, new_dict):
    for key, val in new_dict.items():
        base_dict[key] += val
    new_dict.clear()

def dump_dict_2_hdfs_epoch(merged_dict, column_names, hdfs_path, fs):
    for pid, val in merged_dict.items():
        path = hdfs_path + '/merged'+'/partition_' + str(pid) + '.parquet'
        pdf = pd.DataFrame(val, columns=column_names)
        adf = pa.Table.from_pandas(pdf)
        print(path)
        with fs.open(path,'wb') as f:
            pq.write_table(adf, f,write_statistics=False,use_dictionary=False,compression='none')
    print('= = = exit dumping = = =')


def batch_data_parallel(benchmark,table_name, partition_tree, chunk_size, hdfs_path,hdfs_private_ip):
    begin_time = time.time()
    fs=pa.hdfs.connect(host=hdfs_private_ip, port=port, user='liupengju')
    chunk_count = 0
    print(f'delete existing dirtory:{hdfs_path}/merged')
    if fs.exists(f"{hdfs_path}/merged"):
        fs.delete(path=f"{hdfs_path}/merged",recursive=True)
    table_path=f'{base_dir}/../dataset/{benchmark}/{table_name}.csv'
    base_dict = {}
    for leaf in partition_tree.get_leaves(): base_dict[leaf.nid]=[]
    used_col_names=table_metadata[table_name]['numeric_columns']
    for chunk in pd.read_csv(table_path, usecols=used_col_names, chunksize=chunk_size):
        print('reading chunk: ', chunk_count)
        result_dict=process_chunk(chunk, chunk_count, partition_tree)
        merge_dict(base_dict, result_dict)
        chunk_count += 1
        print('Finsh chunk: ', chunk_count)
    dump_dict_2_hdfs_epoch(base_dict, used_col_names, hdfs_path, fs)
    base_dict.clear()
    finish_time = time.time()
    print('= = = = = TOTAL DATA ROUTING AND PERISITING TIME:', finish_time - begin_time, "= = = = =")


def data_routing(method):
    hyperGraph=None
    opt_time_dict={}
    if method == 'PAW':
        pawGraph=pickle.load(open(f'{base_dir}/../layouts/{benchmark}/paw-hgraph.pkl','rb'))
        qdTreer=pickle.load(open(f'{base_dir}/../layouts/{benchmark}/qd-trees.pkl','rb'))
        paw_opt_time=0
        for table in pawGraph.candidate_nodes.keys():
            start_time=time.time()
            partition_tree=qdTreer[table]
            hdfs_path=f"{hdfs_base_path}/{benchmark}/paw/{table}/"
            batch_data_parallel(benchmark, table, partition_tree, chunk_size, hdfs_path, num_process, hdfs_private_ip)
            end_time=time.time()
            paw_opt_time+=end_time-start_time
        
        opt_time_dict['PAW']=paw_opt_time
            
    else:
        adpGraph=pickle.load(open(f'{base_dir}/../layouts/{benchmark}/adp-hgraph.pkl','rb'))
        joinTreer=pickle.load(open(f'{base_dir}/../layouts/{benchmark}/join-trees.pkl','rb'))

        adp_opt_time,jt_opt_time=0,0
        for table in adpGraph.candidate_nodes.keys():
            for column in adpGraph.candidate_nodes[table].keys():
                start_time=time.time()
                partition_tree=joinTreer[table][column]
                hdfs_path=f"{hdfs_base_path}/{benchmark}/join/{table}/{column}"
                batch_data_parallel(benchmark, table, partition_tree, chunk_size, hdfs_path, num_process, hdfs_private_ip)
                end_time=time.time()

                if column in adpGraph.hyper_nodes[table]:
                    adp_opt_time+=end_time-start_time
                if column in jtGraph.hyper_nodes[table]:
                    jt_opt_time+=end_time-start_time
        opt_time_dict['AD-MTO']=adp_opt_time
        opt_time_dict['PAC-Tree']=jt_opt_time
        
    return opt_time_dict
        

if __name__ == '__main__':
    sc,sqlContext=init_spark()
    opt_time_dict=data_routing()
    print(opt_time_dict)
    sc.stop()