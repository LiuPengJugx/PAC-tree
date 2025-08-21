from partition_algorithm import PartitionAlgorithm
import pickle
import itertools
import time
from join_eval import JoinEvaluator
import os
from join_order import JoinGraph,TraditionalJoinOrder
from join_selector_settings import JoinSelectorSettings
import pandas as pd
import concurrent.futures
from db.conf import table_suffix
import argparse

base_dir=os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--command",type=int, default=0, help="choose the command to run")
parser.add_argument("--benchmark",type=str, default='tpch', help="choose the evaluated benchmark")

parser.add_argument("--init", action='store_true', help="initialize base layouts and join queries")


# python join_key_selector.py --init --benchmark=tpch
# python join_key_selector.py --command=0 --benchmark=tpch

args = parser.parse_args()
settings=JoinSelectorSettings()
settings.benchmark=args.benchmark  #'tpch' imdb

class TwoLayerTree:
    def __init__(self,trees):
        self.candidate_trees=trees

class QdTree:
    def __init__(self,trees):
        self.candidate_trees=trees 

# 超图基类
class BaseHypergraph:
    def __init__(self, benchmark, candidate_nodes):
        self.benchmark = benchmark
        self.candidate_nodes = candidate_nodes
        self.hyper_nodes = {table: [] for table in candidate_nodes}

    def clear_hyper_nodes(self):
        self.hyper_nodes = {table: [] for table in self.candidate_nodes}

    def show_selections(self):
        for table, columns in self.hyper_nodes.items():
            print(f"Hypernode: {table}, Selected Columns: {columns}")

# TPC-H 超图类
class HypergraphTPC(BaseHypergraph):
    def __init__(self):
        candidate_nodes = {
            "nation": ["n_nationkey", "n_regionkey"],
            "region": ["r_regionkey"],
            "customer": ["c_custkey", "c_nationkey"],
            "orders": ["o_orderkey", "o_custkey"],
            "lineitem": ["l_partkey", "l_orderkey", "l_suppkey"],
            "part": ["p_partkey"],
            "supplier": ["s_suppkey", "s_nationkey"],
            "partsupp": ["ps_suppkey", "ps_partkey"]
        }
        super().__init__('tpch', candidate_nodes)

# JOB 超图类
class JobHypergraph(BaseHypergraph):
    def __init__(self):
        candidate_nodes = {
            "aka_name": ["id", "person_id"],
            'aka_title': ["id", "movie_id", "kind_id"],
            "cast_info": ["id", "person_id", "movie_id", "person_role_id", "role_id"],
            "char_name": ["id"],
            "comp_cast_type": ["id"],
            "company_name": ["id"],
            "company_type": ["id"],
            "complete_cast": ["id", "movie_id", "subject_id", "status_id"],
            "info_type": ["id"],
            "keyword": ["id"],
            "kind_type": ["id"],
            "link_type": ["id"],
            "movie_companies": ["id", "movie_id", "company_id", "company_type_id"],
            "movie_info": ["id", "movie_id", "info_type_id"],
            "movie_info_idx": ["id", "movie_id", "info_type_id"],
            "movie_keyword": ["id", "movie_id", "keyword_id"],
            "movie_link": ["link_type_id", "movie_id", "linked_movie_id"],
            "name": ["id"],
            "person_info": ["id", "person_id", "info_type_id"],
            "role_type": ["id"],
            "title": ["id", "kind_id"],
        }
        super().__init__('imdb', candidate_nodes)

# 为所有表创建候选join trees
def create_join_trees(hypergraph):
    tree_dict={}
    partitioner=PartitionAlgorithm(benchmark=hypergraph.benchmark)
    partitioner.load_join_query()
    for table, columns in hypergraph.candidate_nodes.items():
        tree_dict[table]={}
        for column in columns:
            partitioner.table_name=table
            partitioner.load_data()
            partitioner.load_query()
            partitioner.LogicalJoinTree(join_col=column)
            tree_dict[table][column]=partitioner.partition_tree
    return TwoLayerTree(tree_dict)


def create_qdtrees(hypergraph):
    tree_dict={}
    pa=PartitionAlgorithm(benchmark=hypergraph.benchmark)
    pa.load_join_query(is_join_indeuced='PAW')
    for table, _ in hypergraph.candidate_nodes.items():
        pa.table_name=table
        pa.load_data()
        pa.load_query(is_join_indeuced='PAW')
        pa.InitializeWithQDT()
        tree_dict[table]=pa.partition_tree
    return QdTree(tree_dict)

def load_join_queries(hypergraph,is_join_indeuced):
    pa=PartitionAlgorithm(benchmark=hypergraph.benchmark)
    pa.load_join_query(join_indeuced=is_join_indeuced)
    return pa.join_queries



# 直接读取 joinTreer, join_queries等关键信息
def init_partitioner():
    if settings.benchmark=='tpch':
        hypergraph=HypergraphTPC()
    elif settings.benchmark=='imdb':
        hypergraph=JobHypergraph()
    # Create join trees for all tables
    joinTreer = create_join_trees(hypergraph)
    with open(f'{base_dir}/../layouts/{hypergraph.benchmark}/base_trees/join-trees.pkl','wb') as f:
        pickle.dump(joinTreer,f)

    # # Create QD trees for all tables
    qdTreer = create_qdtrees(hypergraph)
    with open(f'{base_dir}/../layouts/{hypergraph.benchmark}/base_trees/qd-trees.pkl','wb') as f:
        pickle.dump(qdTreer,f)
        
    mto_queries = load_join_queries(hypergraph,is_join_indeuced='MTO')
    with open(f'{base_dir}/../layouts/{hypergraph.benchmark}/used_queries/mto-queries.pkl','wb') as f:
        pickle.dump(mto_queries,f)
    
    pac_queries = load_join_queries(hypergraph,is_join_indeuced='PAC')
    with open(f'{base_dir}/../layouts/{hypergraph.benchmark}/used_queries/pac-queries.pkl','wb') as f:
        pickle.dump(pac_queries,f)

    paw_queries = load_join_queries(hypergraph,is_join_indeuced='PAW')
    with open(f'{base_dir}/../layouts/{hypergraph.benchmark}/used_queries/paw-queries.pkl','wb') as f:
        pickle.dump(paw_queries,f)

# ~~~~~~~~loading global variables~~~~~~~~
if args.init:
    init_partitioner()
    exit(0)

joinTreer=pickle.load(open(f'{base_dir}/../layouts/{settings.benchmark}/base_trees/join-trees.pkl','rb'))
qdTreer=pickle.load(open(f'{base_dir}/../layouts/{settings.benchmark}/base_trees/qd-trees.pkl','rb'))
paw_queries=pickle.load(open(f'{base_dir}/../layouts/{settings.benchmark}/used_queries/paw-queries.pkl','rb'))
mto_queries=pickle.load(open(f'{base_dir}/../layouts/{settings.benchmark}/used_queries/mto-queries.pkl','rb'))
pac_queries=pickle.load(open(f'{base_dir}/../layouts/{settings.benchmark}/used_queries/pac-queries.pkl','rb'))

# ~~~~~~~~读取表的部分元数据信息~~~~~~~~~
metadata={}
for table,tree in joinTreer.candidate_trees.items():
    sample_tree=list(tree.values())[0]
    metadata[table]={'row_count':sample_tree.pt_root.node_size,'used_cols':sample_tree.used_columns}    
table_metadata=pickle.load(open(f'{base_dir}/../dataset/{settings.benchmark}/metadata.pkl','rb'))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def estimate_join_cost(hypergraph,current_table):
    shuffle_times,hyper_times={},{}
    join_cost=0
    for q_id, join_query in enumerate(mto_queries):
        for join_op in join_query['join_relations']:
            left_table,left_col_idx=list(join_op.items())[0]
            right_table,right_col_idx=list(join_op.items())[1]
            left_col,right_col=metadata[left_table]["used_cols"][left_col_idx],metadata[right_table]["used_cols"][right_col_idx]
            if left_table!=current_table or right_table!=current_table:
                continue
            left_query,right_query=join_query['vectors'][left_table],join_query['vectors'][right_table]
            is_hyper=True
            if left_col in hypergraph.hyper_nodes[left_table]:
                left_tree=joinTreer.candidate_trees[left_table][left_col]
            else:
                left_tree=list(joinTreer.candidate_trees[left_table].values())[0]
                is_hyper=False
            if right_col in hypergraph.hyper_nodes[right_table]:
                right_tree=joinTreer.candidate_trees[right_table][right_col]
            else:
                right_tree=list(joinTreer.candidate_trees[right_table].values())[0]
                is_hyper=False
            left_cost=sum([left_tree.nid_node_dict[bid].node_size  for bid in left_tree.query_single(left_query)])
            right_cost=sum([right_tree.nid_node_dict[bid].node_size  for bid in right_tree.query_single(right_query)])
            join_cost+=left_cost+right_cost if is_hyper else (left_cost+3*right_cost)
    
    return join_cost

# 现在，我已经给出了每个超节点初步的候选可以构建超图的边结构了。
# 定义函数：可以大致估算超图中，该表所在边的潜在join成本
def processing_join_workload(hypergraph, group_type, join_strategy='traditional', tree_type='paw'):
    shuffle_times, hyper_times = {}, {}
    join_cost_dict = {}
    cost_log = {}
    byte_log = {}
    solved_tables = {}
    input_queries = paw_queries if tree_type == 'paw' else mto_queries if tree_type == 'mto' else pac_queries
    shuffle_group_time, hyper_group_time = [], []
    tot_order_time = 0

    for q_id, join_query in enumerate(input_queries):
        print('processing query:', q_id)
        if _should_skip_query(q_id):
            continue
            
        query_result = _process_single_query(
            q_id,
            hypergraph,
            join_query,
            tree_type,
            join_strategy,
            group_type,
            tot_order_time,
            shuffle_times,
            hyper_times,
            shuffle_group_time,
            hyper_group_time,
            join_cost_dict,
            solved_tables
        )
        
        if query_result:
            cost_log[q_id], byte_log[q_id] = query_result

    return _generate_final_metrics(
        shuffle_times,
        hyper_times,
        join_cost_dict,
        cost_log,
        byte_log,
        solved_tables,
        tot_order_time,
        hyper_group_time,
        shuffle_group_time
    )

# ---------------------- 完整辅助函数 ----------------------
def _should_skip_query(q_id):
    return (settings.benchmark == 'imdb' and q_id == 7) or \
           (settings.benchmark == 'tpch' and q_id == 6)

def _process_single_query(q_id, hypergraph, join_query, tree_type, strategy, group_type,tot_order_time, 
                         shuffle_times, hyper_times, shuffle_group_time, hyper_group_time, join_cost_dict, solved_tables):
    if not join_query['vectors']:
        print('empty query')
        return None

    query_cost_log, query_bytes_log = [], []
    solved_tables[q_id] = set()

    if join_query['join_relations']:
        join_path, order_time = _generate_join_path(hypergraph, join_query, strategy, tree_type)
        tot_order_time += order_time

        join_artifacts = _process_join_operations(
            hypergraph,
            join_query,
            join_path,
            tree_type,
            group_type,
            shuffle_times,
            hyper_times,
            shuffle_group_time, hyper_group_time, query_cost_log, query_bytes_log,
            join_cost_dict,
            solved_tables[q_id]
        )

        _handle_multi_table_join(hypergraph, join_artifacts, query_cost_log, query_bytes_log)
    else:
        _handle_scan_operations(join_query, tree_type, join_cost_dict, solved_tables[q_id], 
                               query_cost_log, query_bytes_log)

    return query_cost_log, query_bytes_log

def _generate_join_path(hypergraph, join_query, strategy, tree_type):
    start_time = time.time()
    
    if strategy == 'traditional':
        path = TraditionalJoinOrder(join_query['join_relations'], metadata).paths
    else:
        scan_block_dict = _build_scan_block_dict(hypergraph, join_query, tree_type)
        path = JoinGraph(scan_block_dict).generate_MST()
    
    return path, time.time() - start_time

def _build_scan_block_dict(hypergraph, join_query, tree_type):
    scan_block_dict = {'card': {}, 'relation': []}
    
    for join_op in join_query['join_relations']:
        left_table, left_col_idx = list(join_op.items())[0]
        right_table, right_col_idx = list(join_op.items())[1]
        
        _process_join_column(
            join_op,
            hypergraph,
            left_table,
            left_col_idx,
            join_query['vectors'][left_table],
            tree_type,
            scan_block_dict
        )
        
        _process_join_column(
            join_op,
            hypergraph,
            right_table,
            right_col_idx,
            join_query['vectors'][right_table],
            tree_type,
            scan_block_dict
        )
    
    return scan_block_dict

def _process_join_column(join_op,hypergraph, table, col_idx, query_vector, tree_type, scan_block_dict):
    col = metadata[table]["used_cols"][col_idx]
    col_key = f'{table}.{col}'
    
    if tree_type == 'paw':
        tree = qdTreer.candidate_trees[table]
    else:
        tree = (joinTreer.candidate_trees[table][col] 
                if col in hypergraph.hyper_nodes[table] 
                else list(joinTreer.candidate_trees[table].values())[0])
    
    scan_block_dict["card"][col_key] = len(tree.query_single(query_vector))
    scan_block_dict["relation"].append([
        col_key,
        f'{list(join_op.items())[1][0]}.{metadata[list(join_op.items())[1][0]]["used_cols"][list(join_op.items())[1][1]]}',
        'Hyper' if tree_type != 'paw' and col in hypergraph.hyper_nodes[table] else 'Shuffle'
    ])

def _process_join_operations(hypergraph, join_query, join_path, tree_type, group_type, 
                            shuffle_times, hyper_times,shuffle_group_time, hyper_group_time, query_cost_log, query_bytes_log, join_cost_dict, solved_tables):
    artifacts = {
        'temp_table_dfs': [],
        'temp_joined_bytes': [],
        'temp_shuffle_ops': []
    }

    for order_id, item in enumerate(join_path):
        if_realhyper=True

        join_ops = [(item[0].table, item[0].adj_col[item[1]]), 
                   (item[1].table, item[1].adj_col[item[0]])]
        
        if item[2] == 1:  # Hyper join
            _handle_hyper_join(
                hypergraph,
                join_ops,
                join_query,
                tree_type,
                group_type,
                artifacts,
                hyper_times,
                shuffle_times,
                shuffle_group_time, hyper_group_time, query_cost_log, query_bytes_log,
                join_cost_dict,
                solved_tables,
                if_realhyper
            )
        else:  # Shuffle join
            _handle_shuffle_join(
                hypergraph,
                join_ops,
                join_query,
                tree_type,
                order_id,
                len(join_path),
                artifacts,
                join_path,
                if_realhyper
            )

    return artifacts

def _handle_hyper_join(hypergraph, join_ops, join_query, tree_type, group_type, 
                      artifacts, hyper_times,shuffle_times,shuffle_group_time, hyper_group_time, query_cost_log, query_bytes_log, join_cost_dict, solved_tables,if_realhyper):
    joined_trees = []
    joined_tables = []
    join_queryset = []
    joined_cols = []

    for table, col in join_ops:
        query = join_query['vectors'][table]
        col_idx = metadata[table]["used_cols"].index(col)
        
        if tree_type == 'paw':
            tree = qdTreer.candidate_trees[table]
        else:
            tree = (joinTreer.candidate_trees[table][col] 
                    if col in hypergraph.hyper_nodes[table] 
                    else list(joinTreer.candidate_trees[table].values())[0])
        
        joined_trees.append(tree)
        joined_tables.append(table)
        join_queryset.append(query)
        joined_cols.append(col_idx)

    join_eval = JoinEvaluator(
        join_queryset,
        joined_cols,
        joined_trees,
        joined_tables,
        settings.block_size,
        table_metadata,
        benchmark=hypergraph.benchmark
    )
    
    cost, bytes, df, group_time = join_eval.compute_total_shuffle_hyper_cost(group_type, True)
    
    artifacts['temp_table_dfs'].append(df)
    artifacts['temp_joined_bytes'].append(
        table_metadata[joined_tables[0]]['read_line'] + 
        table_metadata[joined_tables[1]]['read_line']
    )
    if if_realhyper:
        hyper_times[joined_tables[0]] = hyper_times.get(joined_tables[0], 0) + 1
        hyper_times[joined_tables[1]] = hyper_times.get(joined_tables[1], 0) + 1
        hyper_group_time.append(group_time)
    else:
        shuffle_times[join_ops[0][0]] = shuffle_times.get(join_ops[0][0], 0) + 1
        shuffle_times[join_ops[1][0]] = shuffle_times.get(join_ops[1][0], 0) + 1
        shuffle_group_time.append(group_time)
    solved_tables.update(joined_tables)
    join_cost_dict[joined_tables[0]] = join_cost_dict.get(joined_tables[0], 0) + cost // 2
    join_cost_dict[joined_tables[1]] = join_cost_dict.get(joined_tables[1], 0) + cost // 2
    query_cost_log.append(cost)
    query_bytes_log.append(bytes)


    

def _handle_shuffle_join(hypergraph, join_ops, join_query, tree_type, order_id, total_steps, 
                        artifacts, join_path,if_realhyper):
    table, col = join_ops[0] if order_id == 0 else join_ops[1]
    query = join_query['vectors'][table]
    
    if order_id == total_steps - 1 or (order_id < total_steps -1 and join_path[order_id+1][2] == -1):
        if tree_type == 'paw':
            tree = qdTreer.candidate_trees[table]
        else:
            tree = list(joinTreer.candidate_trees[table].values())[0]
        
        b_ids = tree.query_single(query)
        dataset = [data for b_id in b_ids for data in tree.nid_node_dict[b_id].dataset]
        
        if hypergraph.benchmark == 'imdb':
            columns = [f"{table_suffix[hypergraph.benchmark][table]}_{col}" for col in tree.used_columns]
        else:
            columns = tree.used_columns
        
        artifacts['temp_table_dfs'].append(pd.DataFrame(dataset, columns=columns))
    
    artifacts['temp_shuffle_ops'].append((
        join_ops[0][0], metadata[join_ops[0][0]]["used_cols"][metadata[join_ops[0][0]]["used_cols"].index(join_ops[0][1])],
        join_ops[1][0], metadata[join_ops[1][0]]["used_cols"][metadata[join_ops[1][0]]["used_cols"].index(join_ops[1][1])]
    ))
    artifacts['temp_joined_bytes'].append(table_metadata[join_ops[1][0]]['read_line'])
    

def _handle_multi_table_join(hypergraph, artifacts, query_cost_log, query_bytes_log):
    if len(artifacts['temp_table_dfs']) > 1:
        shuffle_cost = 0
        shuffle_bytes = 0
        last_temp_dfs = artifacts['temp_table_dfs'][0]
        last_temp_line = artifacts['temp_joined_bytes'][0]

        def pandas_hash_join(df_A, join_col_A, df_B, join_col_B):
            if join_col_B in df_A.columns:
                return df_A
            if df_A.shape[0] > df_B.shape[0]:
                df_A, df_B = df_B, df_A
                join_col_A, join_col_B = join_col_B, join_col_A
            df_B = df_B.drop_duplicates(subset=[join_col_B], keep='first')
            return df_A.merge(df_B, how='inner', left_on=join_col_A, right_on=join_col_B)

        for i in range(1, len(artifacts['temp_table_dfs'])):
            shuffle_cost += (artifacts['temp_table_dfs'][i].shape[0] * 3 + last_temp_dfs.shape[0])
            shuffle_bytes += (artifacts['temp_table_dfs'][i].shape[0] * 3 * last_temp_line + 
                             last_temp_dfs.shape[0] * artifacts['temp_joined_bytes'][i])
            
            join_table1, join_col1, join_table2, join_col2 = artifacts['temp_shuffle_ops'][i-1]
            if hypergraph.benchmark == 'imdb':
                join_col1 = f"{table_suffix[hypergraph.benchmark][join_table1]}_{join_col1}"
                join_col2 = f"{table_suffix[hypergraph.benchmark][join_table2]}_{join_col2}"
            
            last_temp_dfs = pandas_hash_join(last_temp_dfs, join_col1, 
                                            artifacts['temp_table_dfs'][i], join_col2)
            last_temp_line = artifacts['temp_joined_bytes'][i] + last_temp_line

        query_cost_log.append(shuffle_cost)
        query_bytes_log.append(shuffle_bytes)

def _handle_scan_operations(join_query, tree_type, join_cost_dict, solved_tables, 
                           query_cost_log, query_bytes_log):
    for table, query_vector in join_query['vectors'].items():
        solved_tables.add(table)
        if tree_type == 'paw':
            tree = qdTreer.candidate_trees[table]
        else:
            tree = list(joinTreer.candidate_trees[table].values())[0]
        
        b_ids = tree.query_single(query_vector)
        scan_cost = sum([tree.nid_node_dict[b_id].node_size for b_id in b_ids])
        
        join_cost_dict[table] = join_cost_dict.get(table, 0) + scan_cost
        query_cost_log.append(scan_cost)
        query_bytes_log.append(table_metadata[table]['read_line'] * scan_cost)

def _generate_final_metrics(shuffle_times, hyper_times, join_cost_dict, cost_log, byte_log,
                          solved_tables, tot_order_time, hyper_group_time, shuffle_group_time):
    query_cost_dict = {}
    query_ratio_dict = {}
    
    for qid in cost_log:
        query_cost_dict[qid] = sum(cost_log[qid])
        total_bytes = sum([sum(table_metadata[table]['width'].values()) * table_metadata[table]['rows'] 
                          for table in solved_tables[qid]])
        query_ratio_dict[qid] = sum(byte_log[qid]) / total_bytes if total_bytes else 0
        print(f"Query {qid} cost log: {cost_log[qid]} | ratio log: {byte_log[qid]}")

    final_join_result = {}
    for table in join_cost_dict:
        final_join_result[table] = {
            'shuffle times': shuffle_times.get(table, 0),
            'hyper times': hyper_times.get(table, 0),
            'join cost': join_cost_dict[table]
        }

    avg_values = {
        'shuffle times': sum(shuffle_times.values())/len(shuffle_times) if shuffle_times else 0,
        'hyper times': sum(hyper_times.values())/len(hyper_times) if hyper_times else 0,
        'join cost': sum(join_cost_dict.values())/len(join_cost_dict) if join_cost_dict else 0
    }
    final_join_result['avg'] = avg_values

    opt_time = [
        tot_order_time,
        sum(hyper_group_time)/len(hyper_group_time) if hyper_group_time else 0,
        sum(shuffle_group_time)/len(shuffle_group_time) if shuffle_group_time else 0
    ]
    
    return final_join_result, query_cost_dict, query_ratio_dict, opt_time


def select_columns_by_PAW(group_type=0, strategy='traditional'):
    # Build hypergraph structure
    hypergraph=None
    if settings.benchmark=='tpch':
        hypergraph = HypergraphTPC() 
    elif settings.benchmark=='imdb':
        hypergraph = JobHypergraph()
    total_opt_time=[]
    tree_opt_time=0
    for table in hypergraph.hyper_nodes:
        tree_opt_time+=qdTreer.candidate_trees[table].build_time
    total_opt_time.append(tree_opt_time)
    final_result, query_costs, query_ratio, group_opt_time = processing_join_workload(hypergraph, group_type, strategy, tree_type='paw')
    total_opt_time=+group_opt_time
    print("Final Column Selection Cost of PAW is: ",sum(list(query_costs.values()))/len(query_costs.keys()))
    print("Average Data Cost Ratio of PAW is: ",sum(list(query_ratio.values()))/len(query_ratio.keys()))
    print("Total Opt Time of PAW is: ",total_opt_time)
    hypergraph.show_selections()
    with open(f'{base_dir}/../layouts/{hypergraph.benchmark}/join_key_selection/paw-hgraph.pkl','wb') as f:
        pickle.dump(hypergraph,f)
    return final_result, query_costs,query_ratio, sum(total_opt_time)
    

def select_columns_by_AD_MTO(group_type=0, strategy='traditional'):
    hypergraph=None
    if settings.benchmark=='tpch':
        hypergraph = HypergraphTPC() 
    elif settings.benchmark=='imdb':
        hypergraph = JobHypergraph()
    total_opt_time=[]
    tree_opt_time=0
    for table in hypergraph.hyper_nodes:
        col=metadata[table]['used_cols'][0]
        if col in hypergraph.candidate_nodes[table]:
            hypergraph.hyper_nodes[table].append(col)
            tree_opt_time+=joinTreer.candidate_trees[table][col].build_time
    total_opt_time.append(tree_opt_time)
    final_result, query_costs, query_ratio, group_opt_time = processing_join_workload(hypergraph, group_type, strategy, tree_type='mto')
    total_opt_time=+group_opt_time
    print("Final Column Selection Cost of AD-MTO is: ",sum(list(query_costs.values()))/len(query_costs.keys()))
    print("Average Data Cost Ratio of AD-MTO is: ",sum(list(query_ratio.values()))/len(query_ratio.keys()))
    print("Total Opt Time of AD-MTO is: ",total_opt_time)
    hypergraph.show_selections()
    with open(f'{base_dir}/../layouts/{hypergraph.benchmark}/join_key_selection/ad-mto-hgraph.pkl','wb') as f:
        pickle.dump(hypergraph,f)

    return final_result, query_costs, query_ratio, sum(total_opt_time)

def select_columns_by_JT(group_type,strategy='traditional'):

    adj_matrix = {}
    for query in pac_queries:
        for join_op in query["join_relations"]:
            join_relations=set()
            for table,col in join_op.items():
                join_relations.add(f'{table}.{metadata[table]["used_cols"][col]}')
            for key1, key2 in itertools.combinations(join_relations, 2):
                if key1 not in adj_matrix:
                    adj_matrix[key1] = {}
                if key2 not in adj_matrix:
                    adj_matrix[key2] = {}
                adj_matrix[key1][key2] = adj_matrix[key1].get(key2, 0) + 1
                adj_matrix[key2][key1] = adj_matrix[key2].get(key1, 0) + 1
    
    hypergraph=None
    if settings.benchmark=='tpch':
        hypergraph = HypergraphTPC() 
    elif settings.benchmark=='imdb':
        hypergraph = JobHypergraph()
    total_opt_time=[]
    for table in hypergraph.candidate_nodes:
        col_freq_dict={}
        for source_col in hypergraph.candidate_nodes[table]:
            if f'{table}.{source_col}' in adj_matrix:
                for key,freq in adj_matrix[f'{table}.{source_col}'].items():
                    targe_table=key.split('.')[0]
                    tot_freq=freq*table_metadata[targe_table]['rows']
            else:
                tot_freq=0
            col_freq_dict[source_col]=tot_freq
    
        col_list=sorted(col_freq_dict.items(), key=lambda x: x[1], reverse=True)
    
        for i in range(1,settings.replication_factor+1):
            if i > len(col_list):
                break
            hypergraph.hyper_nodes[table].append(col_list[i-1][0])
    tree_opt_time=0
    for table in hypergraph.hyper_nodes:
        for col in hypergraph.hyper_nodes[table]:
            tree_opt_time+=joinTreer.candidate_trees[table][col].build_time
    total_opt_time.append(tree_opt_time)
    final_result, query_costs, query_ratio,group_opt_time = processing_join_workload(hypergraph, group_type, strategy, tree_type='pac')
    total_opt_time+=group_opt_time
    print("Final Column Selection Cost of PAC-Tree is: ",sum(list(query_costs.values()))/len(query_costs.keys()))
    print("Average Data Cost Ratio of PAC-Tree is: ",sum(list(query_ratio.values()))/len(query_ratio.keys()))
    print("Total Opt Time (tree,order,group) of PAC-Tree is: ",total_opt_time)
    hypergraph.show_selections()
    with open(f'{base_dir}/../layouts/{hypergraph.benchmark}/join_key_selection/jt-hgraph.pkl','wb') as f:
        pickle.dump(hypergraph,f)
    return final_result, query_costs, query_ratio, sum(total_opt_time)

def save_experiment_result(experiment_result,disabled_prim_reorder):
    import pandas as pd
    import os

    base_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = f"{base_dir}/../experiment/result/sf={settings.scale_factor}/bs={settings.block_size}/rep={settings.replication_factor}/"
    query_list = []
    table_list = []
    time_overhead_list = []
    for model_name, (table_info, query_info, query_ratio_info, opt_time) in experiment_result.items():
        for q_id, cost_val in query_info.items():
            query_list.append([model_name, q_id, cost_val,query_ratio_info[q_id]])
        for table, cost_dict in table_info.items():
            shuffle_times = cost_dict.get('shuffle times', 0)
            hyper_times = cost_dict.get('hyper times', 0)
            join_cost = cost_dict.get('join cost', 0)
            table_list.append([model_name, table, shuffle_times, hyper_times, join_cost])
        time_overhead_list.append([model_name, opt_time])

    df_query = pd.DataFrame(query_list, columns=["Model", "Query ID", "Join Cost", "Scan Ratio"])
    df_table = pd.DataFrame(table_list, columns=["Model", "Table", "Shuffle Times", "Hyper Times", "Join Cost"])
    df_time = pd.DataFrame(time_overhead_list, columns=["Model", "Opt Time"])

    # pivot_query = df_query.pivot(index="Query ID", columns="Model", values="Join Cost").reset_index()
    pivot_query_cost = df_query.pivot_table(index="Query ID", columns="Model", values="Join Cost", aggfunc="first").reset_index()
    pivot_ratio = df_query.pivot_table(index="Query ID", columns="Model", values="Scan Ratio", aggfunc="first").reset_index()
    pivot_query_cost.rename(columns={
        "PAW": "PAW_Join Cost",
        "ADP": "AD-MTO_Join Cost",
        "JT": "PAC-Tree_Join Cost"
    }, inplace=True)
    
    pivot_ratio.rename(columns={
        "PAW": "PAW_Scan Ratio",
        "ADP": "AD-MTO_Scan Ratio",
        "JT": "PAC-Tree_Scan Ratio"
    }, inplace=True)
    merged_query_df = pd.merge(pivot_query_cost, pivot_ratio, on="Query ID")
    
    
    pivot_shuffle = df_table.pivot_table(index="Table", columns="Model", values="Shuffle Times", aggfunc="first").reset_index()
    pivot_hyper = df_table.pivot_table(index="Table", columns="Model", values="Hyper Times", aggfunc="first").reset_index()
    pivot_cost = df_table.pivot_table(index="Table", columns="Model", values="Join Cost", aggfunc="first").reset_index()
    pivot_shuffle.rename(
        columns={
            "PAW": "PAW_shuffle_times",
            "ADP": "AD-MTO_shuffle_times",
            "JT": "PAC-Tree_shuffle_times",
        },
        inplace=True
    )
    pivot_hyper.rename(
        columns={
            "PAW": "PAW_hyper_times",
            "ADP": "AD-MTO_hyper_times",
            "JT": "PAC-Tree_hyper_times"
        },
        inplace=True
    )
    merged_df = pivot_shuffle.merge(pivot_hyper, on="Table")
    merged_df = merged_df.merge(pivot_cost, on="Table", suffixes=(None, "_cost"))
    merged_df.rename(
        columns={
            "PAW": "PAW_Scan Block",
            "ADP": "AD-MTO_Scan_Block",
            "JT": "PAC-Tree_Scan_Block",
        },
        inplace=True
    ) 
    

    os.makedirs(result_dir, exist_ok=True)
    res_saved_path=f"{result_dir}/{settings.benchmark}_results.xlsx" if disabled_prim_reorder else f"{result_dir}/{settings.benchmark}_results_enable_prim.xlsx"

    with pd.ExcelWriter(res_saved_path) as writer:
        merged_query_df.to_excel(writer, sheet_name="Query_Cost", index=False)
        merged_df.to_excel(writer, sheet_name="Table_Cost", index=False)
        df_time.to_excel(writer, sheet_name="Time_Cost", index=False)




def run_evaluation(save_result=False, disabled_prim_reorder=False):
    """
    disabled_prim_reorder: 是否使用 基于prim算法的join reorder。若为True，则不使用join reorder
    """
    
    experiment_data = {}
    # 能不能改成多线程，这样可以同时运行多个实验

    # 配置日志
    # logging.basicConfig(filename=f'{base_dir}/../logs/column_selection_experiment.log',
    #                     level=logging.INFO,
    #                     format='%(asctime)s - %(levelname)s - %(message)s')

    using_join_order_strategy='traditional' if disabled_prim_reorder else 'prim'   # MTO的prim形式才能显示真实的计划
    def run_PAW():
        res_PAW, cost_PAW, ratio_PAW, total_opt_time = select_columns_by_PAW(strategy=using_join_order_strategy, group_type=0) 
        return [res_PAW, cost_PAW, ratio_PAW, total_opt_time]
    def run_AD_MTO():
        res_ADP, cost_ADP, ratio_ADP, total_opt_time = select_columns_by_AD_MTO(strategy=using_join_order_strategy, group_type=0)
        return [res_ADP, cost_ADP, ratio_ADP, total_opt_time]
        
    def run_JT():
        res_JT, cost_JT,ratio_JT, total_opt_time = select_columns_by_JT(group_type=1, strategy='prim')
        return [res_JT, cost_JT, ratio_JT, total_opt_time]
    
    run_PAW()
    # run_AD_MTO()
    # run_JT()
    exit(-1)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_paw = executor.submit(run_PAW)
        future_adp = executor.submit(run_AD_MTO)
        future_jt  = executor.submit(run_JT)

        result_paw = future_paw.result()
        result_adp = future_adp.result()
        result_jt  = future_jt.result()
        
        experiment_data["PAW"]=result_paw
        experiment_data["ADP"]=result_adp
        experiment_data["JT"]=result_jt
    print(experiment_data)
    
    if save_result:
        save_experiment_result(experiment_data, disabled_prim_reorder) 


def scaling_block_size():
    for item in [1000, 5000, 50000, 100000, 20000, 500000]:
        settings.block_size=item
        run_evaluation(save_result=True, disabled_prim_reorder=True)

def scaling_replication():
    # for item in [1,3,4,5]:
    for item in [1,2,3]:
        settings.replication_factor=item
        run_evaluation(save_result=True, disabled_prim_reorder=True)

def scaling_data_factor():
    # 本质上，也是根据scale-factor来调整block-size的大小
    # 1G=>10000
    # 10G=>1000  因为我目前只有1G数据， 通过缩放来判断。 但是访问的每个块大小要乘以系数  scaling_ratio
    for sf in [10,50,100,200]:
        settings.scale_factor=sf
        scaling_ratio=sf/1
        settings.block_size=10000/scaling_ratio
        run_evaluation(save_result=True, disabled_prim_reorder=True)
        # 将最终结果： 块字节数*scaling_ratio 作为真实的块字节数

if args.command == 0:  #基础实验
    run_evaluation(save_result=False, disabled_prim_reorder=True)
elif args.command == 1: # 扩展实验
    scaling_block_size()
elif args.command == 2:
    scaling_replication()
elif args.command == 3:
    scaling_data_factor()
else:
    print("Invalid command")


# 待做项：
# 1. 优化prim算法，主要是考虑加入更多的初始节点（定义更多的启发式规则）
# (已解决) 2.检查一下opt time的构成，目前看，为什么paw的时间成本竟然高于mto…（难道join树构建更快？） ，另外，group time是不是目前影响较小？？？
# (已解决) 3.opt time是不是要分成两部分：树构建时间和 查询时的search和group时间，才更加合理一些？


"""
问题3：测试详情，对其进行分析即可。可以看出，就group time的对比，PAC-Tree仅为MTO的两倍，平均每个查询后，基本可以忽略不计。

TPC-H: SF=1, RF=1
#tree opt time, group opt time
PAW [220.62583136558533, 1.317516803741455]
MTO [77.79238700866699, 0.19784927368164062]  0.1978/20条查询=0.0099
PAC-Tree [77.79238700866699, 0.36261582374572754]  0.362/20条查询=0.0181
(对于耗时长的查询，可忽略不计)

TPC-H: SF=1, RF=2
#tree opt time, group opt time
PAW [220.62583136558533, 1.3038005828857422]
MTO [77.79238700866699, 0.192124605178833] 
PAC-Tree [201.8400537967682, 0.39661192893981934] 

TPC-H: SF=1, RF=3
#tree opt time, group opt time
PAW [220.62583136558533, 1.3038005828857422]
MTO [77.79238700866699, 0.192124605178833] 
PAC-Tree [332.2015097141266, 0.39661192893981934] 
"""

"""
TPC-H: SF=1, RF=1
#tree opt time, group opt time
PAW [16.26471972465515, 205.2319917678833] （因为shuffle阶段较多？所以group time不应该考虑shuffle开销吧？？？这个本来也是 hyper join的优势，因为shuffle树在对齐中浪费了太多的资源，且效果一般）
[16.487749576568604, 184.51953601837158]
[26.774851083755493, [0.0475921630859375, 3.5144803524017334]]
"""
