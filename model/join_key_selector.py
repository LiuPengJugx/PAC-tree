from partition_algorithm import PartitionAlgorithm
import pickle
import itertools
import time
from join_eval import JoinEvaluator
import os
from join_order import JoinGraph,TraditionalJoinOrder
from conf.context import Context
import pandas as pd
import concurrent.futures
from db.conf import table_suffix
import argparse
from rich.console import Console
from rich.logging import RichHandler
import logging

base_dir=os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--command",type=int, default=0, help="choose the command to run")
parser.add_argument("--benchmark",type=str, default='tpcds', help="choose the evaluated benchmark")
parser.add_argument("--mode", type=str, default='debug', help="choose the mode to run")
parser.add_argument("--init", type=bool, default=False, help="initialize base layouts and join queries")

args = parser.parse_args()
settings=Context()
settings.benchmark=args.benchmark  #'tpch' imdb
# settings.block_size=200  # 5000, 20000,50000

class Debugger:
    def __init__(self):
        self.join_cost_distributions = {}
        self.query_cost_distributions = {}


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

# TPC-DS 超图类
class HypergraphTPCDS(BaseHypergraph):
    def __init__(self):
        candidate_nodes = {
            # Tables that have joins in queries
            "catalog_sales": ["cs_bill_customer_sk", "cs_ship_customer_sk", "cs_sold_date_sk", 
                             "cs_item_sk", "cs_bill_cdemo_sk"],
            "customer": ["c_customer_sk", "c_current_addr_sk", "c_current_cdemo_sk"],
            "customer_address": ["ca_address_sk"],
            "customer_demographics": ["cd_demo_sk"],
            "date_dim": ["d_date_sk"],
            "household_demographics": ["hd_demo_sk"],
            "inventory": ["inv_date_sk", "inv_item_sk"],
            "item": ["i_item_sk"],
            "promotion": ["p_promo_sk"],
            "store": ["s_store_sk"],
            "store_returns": ["sr_customer_sk", "sr_item_sk", "sr_ticket_number", 
                             "sr_returned_date_sk"],
            "store_sales": ["ss_sold_date_sk", "ss_item_sk", "ss_customer_sk", 
                           "ss_cdemo_sk", "ss_hdemo_sk", "ss_addr_sk", 
                           "ss_store_sk", "ss_promo_sk", "ss_ticket_number"],
            "web_sales": ["ws_bill_customer_sk", "ws_item_sk", "ws_sold_date_sk"],
            # Tables that don't have joins in current query set, selecting their most likely join keys
            "call_center": ["cc_call_center_sk"],  # Primary key
            "catalog_page": ["cp_catalog_page_sk"],  # Primary key
            "catalog_returns": ["cr_item_sk", "cr_order_number", "cr_returned_date_sk"],  # Composite primary key and common foreign keys
            "income_band": ["ib_income_band_sk"],  # Primary key
            "reason": ["r_reason_sk"],  # Primary key
            "ship_mode": ["sm_ship_mode_sk"],  # Primary key
            "time_dim": ["t_time_sk"],  # Primary key
            "warehouse": ["w_warehouse_sk"],  # Primary key
            "web_page": ["wp_web_page_sk"],  # Primary key
            "web_returns": ["wr_item_sk", "wr_order_number", "wr_returned_date_sk"],  # Composite primary key and common foreign keys
            "web_site": ["web_site_sk"]  # Primary key
        }
        super().__init__('tpcds', candidate_nodes)

def create_join_trees(hypergraph):
    tree_dict={}
    partitioner=PartitionAlgorithm(benchmark=hypergraph.benchmark,block_size=settings.block_size)
    partitioner.load_join_query()
    for table, columns in hypergraph.candidate_nodes.items():
        tree_dict[table]={}
        partitioner.table_name=table
        partitioner.load_data()
        partitioner.load_query()
        for column in columns:
            partitioner.LogicalJoinTree(join_col=column)
            tree_dict[table][column]=partitioner.partition_tree
    return TwoLayerTree(tree_dict)

def create_color_logger(name='color_logger', level=logging.INFO):
    console = Console()
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)]
    )
    return logging.getLogger(name)

def create_qdtrees(hypergraph):
    tree_dict={}
    pa=PartitionAlgorithm(benchmark=hypergraph.benchmark,block_size=settings.block_size)
    pa.load_join_query(join_indeuced='PAW')
    for table, _ in hypergraph.candidate_nodes.items():
        pa.table_name=table
        pa.load_data()
        pa.load_query(join_indeuced='PAW')
        pa.InitializeWithQDT()
        tree_dict[table]=pa.partition_tree
    return QdTree(tree_dict)

def load_join_queries(hypergraph,is_join_indeuced):
    pa=PartitionAlgorithm(benchmark=hypergraph.benchmark,block_size=settings.block_size)
    pa.load_join_query(join_indeuced=is_join_indeuced)
    return pa.join_queries


dubug_logger=Debugger()
logger = create_color_logger()

def init_partitioner():
    if settings.benchmark=='tpch':
        hypergraph=HypergraphTPC()
    elif settings.benchmark=='imdb':
        hypergraph=JobHypergraph()
    elif settings.benchmark=='tpcds':
        hypergraph=HypergraphTPCDS()
    # Create join trees for all tables
    joinTreer = create_join_trees(hypergraph)
    join_trees_dir = f'{base_dir}/../layouts/bs={settings.block_size}/{hypergraph.benchmark}/base_trees'
    os.makedirs(join_trees_dir, exist_ok=True)
    with open(f'{join_trees_dir}/join-trees.pkl', 'wb') as f:
        pickle.dump(joinTreer, f)

    qdTreer = create_qdtrees(hypergraph)
    qd_trees_dir = f'{base_dir}/../layouts/bs={settings.block_size}/{hypergraph.benchmark}/base_trees'
    os.makedirs(qd_trees_dir, exist_ok=True)
    with open(f'{qd_trees_dir}/qd-trees.pkl', 'wb') as f:
        pickle.dump(qdTreer, f)

def init_workload():
    if settings.benchmark=='tpch':
        hypergraph=HypergraphTPC()
    elif settings.benchmark=='imdb':
        hypergraph=JobHypergraph()
    elif settings.benchmark=='tpcds':
        hypergraph=HypergraphTPCDS()
      
    queries_base_dir = f'{base_dir}/../layouts/bs={settings.block_size}/{hypergraph.benchmark}/used_queries'
    os.makedirs(queries_base_dir, exist_ok=True)

    paw_queries = load_join_queries(hypergraph, is_join_indeuced='PAW')
    paw_queries_path = os.path.join(queries_base_dir, 'paw-queries.pkl')
    with open(paw_queries_path, 'wb') as f:
        pickle.dump(paw_queries, f)

    mto_queries = load_join_queries(hypergraph, is_join_indeuced='MTO')
    mto_queries_path = os.path.join(queries_base_dir, 'mto-queries.pkl')
    with open(mto_queries_path, 'wb') as f:
        pickle.dump(mto_queries, f)

    pac_queries = load_join_queries(hypergraph, is_join_indeuced='PAC')
    pac_queries_path = os.path.join(queries_base_dir, 'pac-queries.pkl')
    with open(pac_queries_path, 'wb') as f:
        pickle.dump(pac_queries, f)


# ~~~~~~~~loading global variables~~~~~~~~
if args.init:
    init_workload()
    init_partitioner()
    exit(0)
    
# if args.mode=='debug':
#     init_workload()

class PartitionContext:
    def __init__(self):
        self.joinTreer = None
        self.qdTreer = None
        self.paw_queries = None
        self.mto_queries = None
        self.pac_queries = None
        self.metadata = None
        self.table_metadata = None

context=PartitionContext()

def load_tree_context():
    context.joinTreer=pickle.load(open(f'{base_dir}/../layouts/bs={settings.block_size}/{settings.benchmark}/base_trees/join-trees.pkl','rb'))
    context.qdTreer=pickle.load(open(f'{base_dir}/../layouts/bs={settings.block_size}/{settings.benchmark}/base_trees/qd-trees.pkl','rb'))
    context.paw_queries=pickle.load(open(f'{base_dir}/../layouts/bs={settings.block_size}/{settings.benchmark}/used_queries/paw-queries.pkl','rb'))
    context.mto_queries=pickle.load(open(f'{base_dir}/../layouts/bs={settings.block_size}/{settings.benchmark}/used_queries/mto-queries.pkl','rb'))
    context.pac_queries=pickle.load(open(f'{base_dir}/../layouts/bs={settings.block_size}/{settings.benchmark}/used_queries/pac-queries.pkl','rb'))

    # ~~~~~~~~Load partial table metadata information~~~~~~~~
    context.metadata={}
    for table,tree in context.joinTreer.candidate_trees.items():
        sample_tree=list(tree.values())[0]
        context.metadata[table]={'row_count':sample_tree.pt_root.node_size,'used_cols':sample_tree.used_columns}    
    context.table_metadata=pickle.load(open(f'{base_dir}/../dataset/{settings.benchmark}/metadata.pkl','rb'))
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def estimate_join_cost(hypergraph,current_table):
    shuffle_times,hyper_times={},{}
    join_cost=0
    for q_id, join_query in enumerate(context.mto_queries):
        for join_op in join_query['join_relations']:
            left_table,left_col_idx=list(join_op.items())[0]
            right_table,right_col_idx=list(join_op.items())[1]
            left_col,right_col=context.metadata[left_table]["used_cols"][left_col_idx],context.metadata[right_table]["used_cols"][right_col_idx]
            if left_table!=current_table or right_table!=current_table:
                continue
            left_query,right_query=join_query['vectors'][left_table],join_query['vectors'][right_table]
            is_hyper=True
            if left_col in hypergraph.hyper_nodes[left_table]:
                left_tree=context.joinTreer.candidate_trees[left_table][left_col]
            else:
                left_tree=list(context.joinTreer.candidate_trees[left_table].values())[0]
                is_hyper=False
            if right_col in hypergraph.hyper_nodes[right_table]:
                right_tree=context.joinTreer.candidate_trees[right_table][right_col]
            else:
                right_tree=list(context.joinTreer.candidate_trees[right_table].values())[0]
                is_hyper=False
            left_cost=sum([left_tree.nid_node_dict[bid].node_size  for bid in left_tree.query_single(left_query)])
            right_cost=sum([right_tree.nid_node_dict[bid].node_size  for bid in right_tree.query_single(right_query)])
            join_cost+=left_cost+right_cost if is_hyper else (left_cost+3*right_cost)
    
    return join_cost

def _should_skip_query(q_id):
    # skip query 7 in imdb and query 6 in tpch (invalid ids)
    return (settings.benchmark == 'imdb' and q_id == 7) or \
           (settings.benchmark == 'tpch' and q_id == 6) or \
           (settings.benchmark == 'tpcds' and q_id == -1)

def processing_join_workload(hypergraph, group_type, join_strategy='traditional', tree_type='paw'):
    """
    Process join workload and calculate the positions and total cost of hyper edges and shuffle edges in the current hypergraph
    
    Args:
        hypergraph: Hypergraph structure
        group_type: Group type (0 or 1)
        join_strategy: Join strategy ('traditional' or 'prim')
        tree_type: Tree type ('paw', 'mto' or 'pac')
        
    Returns:
        final_join_result: Table-level join result statistics
        query_cost_dict: Query-level cost statistics
        query_ratio_dict: Query-level ratio statistics
        opt_time: Optimization time statistics
    """
    # Initialize statistics trackers
    shuffle_times, hyper_times = {}, {}
    join_cost_dict = {}
    cost_log, byte_log = {}, {}
    solved_tables = {}
    shuffle_group_time, hyper_group_time = [], []
    tot_order_time = 0
    
    # Select appropriate query set
    input_queries = context.paw_queries if tree_type == 'paw' else \
                  context.mto_queries if tree_type == 'mto' else context.pac_queries
    
    # ========== Process each query ==========
    for q_id, join_query in enumerate(input_queries):
        print('processing query:', q_id)

        if _should_skip_query(q_id) or join_query['vectors'] == {}:
            if join_query['vectors'] == {}:
                print(f'empty query : {settings.benchmark} {q_id}')
            continue
        
        query_cost_log, query_bytes_log = [], []
        solved_tables[q_id] = set()
        
        # ---------- Process queries without join relations ----------
        if not join_query['join_relations']:
            print('no join operation')
            for table, query_vector in join_query['vectors'].items():
                solved_tables[q_id].add(table)
                
                # Select appropriate tree
                if tree_type == 'paw':
                    tree = context.qdTreer.candidate_trees[table]
                else:
                    tree = list(context.joinTreer.candidate_trees[table].values())[0]
                
                # Calculate scan cost
                b_ids = tree.query_single(query_vector)
                scan_cost = sum([tree.nid_node_dict[b_id].node_size for b_id in b_ids])
                join_cost_dict[table] = join_cost_dict.get(table, 0) + scan_cost
                query_cost_log.append(scan_cost)
                query_bytes_log.append(context.table_metadata[table]['read_line'] * scan_cost)
                logger.info("Just scan table: %s, cost: %s", table, scan_cost)
                
            cost_log[q_id] = query_cost_log
            byte_log[q_id] = query_bytes_log
            continue
        
        # ---------- Process queries with join relations ----------
        # 1. Determine join order
        start1_time = time.time()
        join_path = []
        
        if join_strategy == 'traditional':
            join_path = TraditionalJoinOrder(join_query['join_relations'], context.metadata).paths
        else:
            # Build scan block dictionary for Prim algorithm
            scan_block_dict = {'card': {}, 'relation': []}
            
            for join_op in join_query['join_relations']:
                left_table, left_col_idx = list(join_op.items())[0]
                right_table, right_col_idx = list(join_op.items())[1]
                left_col = context.metadata[left_table]["used_cols"][left_col_idx]
                right_col = context.metadata[right_table]["used_cols"][right_col_idx]
                left_query = join_query['vectors'][left_table]
                right_query = join_query['vectors'][right_table]
                is_hyper = True
                
                # Process PAW tree type
                if tree_type == 'paw':
                    if f'{left_table}.{left_col}' not in scan_block_dict["card"]:
                        scan_block_dict["card"][f'{left_table}.{left_col}'] = len(context.qdTreer.candidate_trees[left_table].query_single(left_query))
                    
                    if f'{right_table}.{right_col}' not in scan_block_dict["card"]:
                        scan_block_dict["card"][f'{right_table}.{right_col}'] = len(context.qdTreer.candidate_trees[right_table].query_single(right_query))
                    is_hyper = False
                # Process MTO or PAC tree type
                else:
                    # Process left table tree
                    if left_col in hypergraph.hyper_nodes[left_table]:
                        tree = context.joinTreer.candidate_trees[left_table][left_col]
                    else:
                        tree = list(context.joinTreer.candidate_trees[left_table].values())[0]
                        is_hyper = False
                    
                    if f'{left_table}.{left_col}' not in scan_block_dict["card"]:
                        scan_block_dict["card"][f'{left_table}.{left_col}'] = len(tree.query_single(left_query))
                    
                    # Process right table tree
                    if right_col in hypergraph.hyper_nodes[right_table]:
                        tree = context.joinTreer.candidate_trees[right_table][right_col]
                    else:
                        tree = list(context.joinTreer.candidate_trees[right_table].values())[0]
                        is_hyper = False
                    
                    if f'{right_table}.{right_col}' not in scan_block_dict["card"]:
                        scan_block_dict["card"][f'{right_table}.{right_col}'] = len(tree.query_single(right_query))
                
                scan_block_dict["relation"].append([
                    f'{left_table}.{left_col}',
                    f'{right_table}.{right_col}',
                    'Hyper' if is_hyper else 'Shuffle'
                ])
            
            # Generate MST using JoinGraph
            jg = JoinGraph(scan_block_dict,hypergraph.hyper_nodes)
            join_path = jg.generate_MST()
        
        tot_order_time += time.time() - start1_time  # Calculate join reorder time
        
        # 2. Process join path
        temp_table_dfs = []
        temp_joined_bytes = []
        temp_shuffle_ops = []
        
        # Iterate through each item in join path
        for order_id, item in enumerate(join_path):
            join_queryset, joined_cols, joined_trees, joined_tables = [], [], [], []
            is_hyper = True if item[2] == 1 else False
            
            is_real_hyper = True
            join_ops = [
                (item[0].table, item[0].adj_col[item[1]]),
                (item[1].table, item[1].adj_col[item[0]])
            ]
            
            # Collect join operation information
            for join_table, join_col in join_ops:
                join_queryset.append(join_query['vectors'][join_table])
                join_col_idx = context.metadata[join_table]["used_cols"].index(join_col)
                joined_cols.append(join_col_idx)
                
                # Select appropriate tree
                if tree_type == 'paw':
                    tree = context.qdTreer.candidate_trees[join_table]
                    is_real_hyper = False
                else:
                    if join_col in hypergraph.hyper_nodes[join_table]:
                        tree = context.joinTreer.candidate_trees[join_table][join_col]
                    else:
                        tree = list(context.joinTreer.candidate_trees[join_table].values())[0]
                        is_real_hyper = False
                
                joined_trees.append(tree)
                joined_tables.append(join_table)
            
            # Process shuffle join
            if not is_hyper:
                cur_idx = 0 if order_id == 0 else 1
                
                # Current is last path or next is not hyper
                if order_id == len(join_path) - 1 or join_path[order_id+1][2] == -1:
                    tree = joined_trees[cur_idx]
                    cur_table = joined_tables[cur_idx]
                    solved_tables[q_id].add(cur_table)
                    
                    # Get table data
                    b_ids = tree.query_single(join_queryset[cur_idx])
                    table_dataset = []
                    for b_id in b_ids:
                        table_dataset += list(tree.nid_node_dict[b_id].dataset)
                    
                    # Set column names
                    if hypergraph.benchmark == 'imdb':
                        used_columns = [table_suffix[hypergraph.benchmark][cur_table]+"_"+col for col in tree.used_columns]
                    else:
                        used_columns = tree.used_columns
                    
                    temp_table_dfs.append(pd.DataFrame(table_dataset, columns=used_columns))
                
                temp_shuffle_ops.append((
                    item[0].table, item[0].adj_col[item[1]], 
                    item[1].table, item[1].adj_col[item[0]]
                ))
                temp_joined_bytes.append(context.table_metadata[item[1].table]['read_line'])
                continue
            
            # Process hyper join
            # Ensure smaller table is on the left (optimization)
            if len(joined_trees[0].query_single(join_queryset[0])) > len(joined_trees[1].query_single(join_queryset[1])):
                join_queryset[0], join_queryset[1] = join_queryset[1], join_queryset[0]
                joined_cols[0], joined_cols[1] = joined_cols[1], joined_cols[0]
                joined_trees[0], joined_trees[1] = joined_trees[1], joined_trees[0]
                joined_tables[0], joined_tables[1] = joined_tables[1], joined_tables[0]
            
            left_table, right_table = joined_tables[0], joined_tables[1]
            
            # Update statistics
            if is_real_hyper:
                hyper_times[left_table] = hyper_times.get(left_table, 0) + 1
                hyper_times[right_table] = hyper_times.get(right_table, 0) + 1
            else:
                shuffle_times[left_table] = shuffle_times.get(left_table, 0) + 1
                shuffle_times[right_table] = shuffle_times.get(right_table, 0) + 1
            
            solved_tables[q_id].add(left_table)
            solved_tables[q_id].add(right_table)
            
            # Execute join evaluation
            join_eval = JoinEvaluator(
                join_queryset, joined_cols, joined_trees, joined_tables,
                settings.block_size, context.table_metadata, benchmark=hypergraph.benchmark
            )
            hyper_shuffle_cost, hyper_shuffle_read_bytes, temp_joined_df, group_time = join_eval.compute_total_shuffle_hyper_cost(group_type, is_real_hyper)
            
            # Update statistics
            join_cost_dict[left_table] = join_cost_dict.get(left_table, 0) + hyper_shuffle_cost // 2
            join_cost_dict[right_table] = join_cost_dict.get(right_table, 0) + hyper_shuffle_cost // 2
            if is_real_hyper:
                hyper_group_time.append(group_time)
            else:
                shuffle_group_time.append(group_time)
            query_cost_log.append(hyper_shuffle_cost)
            query_bytes_log.append(hyper_shuffle_read_bytes)
            logger.info("Join %s and %s, %s hyper cost: %s", left_table, right_table, '' if is_real_hyper else 'fake',hyper_shuffle_cost)
            temp_table_dfs.append(temp_joined_df)
            temp_joined_bytes.append(context.table_metadata[left_table]['read_line'] +context.table_metadata[right_table]['read_line'])
        
        # 3. Process multi-table join
        if len(temp_table_dfs) > 1:
            query_cost_log.append(-1)  # Mark start of multi-table join
            shuffle_cost = 0
            shuffle_bytes = 0
            last_temp_dfs = temp_table_dfs[0]
            last_temp_line = temp_joined_bytes[0]
            
            # Define hash join function
            def pandas_hash_join(df_A, join_col_A, df_B, join_col_B):
                if join_col_B in df_A.columns:
                    return df_A
                if df_A.shape[0] > df_B.shape[0]:
                    df_A, df_B = df_B, df_A
                    join_col_A, join_col_B = join_col_B, join_col_A
                df_B = df_B.drop_duplicates(subset=[join_col_B], keep='first')
                merged_df = df_A.merge(df_B, how='inner', left_on=join_col_A, right_on=join_col_B)
                return merged_df
            
            # Execute multi-table join
            for i in range(1, len(temp_table_dfs)):
                
                join_table1, join_col1, join_table2, join_col2 = temp_shuffle_ops[i-1]
                
                sample_rate1,sample_rate2 = 1, 1

                if join_col1 in context.joinTreer.candidate_trees[join_table1] and hasattr(context.joinTreer.candidate_trees[join_table1][join_col1], 'sample_rate'):
                    sample_rate1=context.joinTreer.candidate_trees[join_table1][join_col1].sample_rate

                if join_col2 in context.joinTreer.candidate_trees[join_table2] and hasattr(context.joinTreer.candidate_trees[join_table2][join_col2], 'sample_rate'):
                    sample_rate2=context.joinTreer.candidate_trees[join_table2][join_col2].sample_rate
                
                data_sample_rate=min(sample_rate1,sample_rate2)
                
                # Calculate cost
                shuffle_cost += int((temp_table_dfs[i].shape[0] * 3 + last_temp_dfs.shape[0]* 1)/data_sample_rate)
                shuffle_bytes += int((temp_table_dfs[i].shape[0] * 3 * last_temp_line + 
                                 last_temp_dfs.shape[0]* 1 * temp_joined_bytes[i])/data_sample_rate)
                
                if hypergraph.benchmark == 'imdb':
                    join_col1 = table_suffix[hypergraph.benchmark][join_table1] + "_" + join_col1
                    join_col2 = table_suffix[hypergraph.benchmark][join_table2] + "_" + join_col2
                
                # Execute join
                last_temp_dfs = pandas_hash_join(last_temp_dfs, join_col1, temp_table_dfs[i], join_col2)
                last_temp_line = temp_joined_bytes[i] + last_temp_line
                
                # Update statistics
                shuffle_times[join_table1] = shuffle_times.get(join_table1, 0) + 1
                shuffle_times[join_table2] = shuffle_times.get(join_table2, 0) + 1
                join_cost_dict[join_table1] = join_cost_dict.get(join_table1, 0) + shuffle_cost // 2
                join_cost_dict[join_table2] = join_cost_dict.get(join_table2, 0) + shuffle_cost // 2
                
                logger.info("Join %s and %s, shuffle cost: %s", join_table1, join_table2, shuffle_cost)
            
            query_cost_log.append(shuffle_cost)
            query_bytes_log.append(shuffle_bytes)
        
        cost_log[q_id] = query_cost_log
        byte_log[q_id] = query_bytes_log
    
    # ========== Summarize results ==========
    # 1. Calculate query costs and ratios
    query_cost_dict = {}
    query_ratio_dict = {}
    
    for qid in cost_log.keys():
        query_cost_dict[qid] = sum(list(cost_log[qid]))
        query_ratio_dict[qid] = sum(list(byte_log[qid])) / sum(
            [sum(context.table_metadata[table]['width'].values()) * context.table_metadata[table]['rows'] 
             for table in solved_tables[qid]]
        )
        print(f"Query {qid} cost log: {cost_log[qid]}  |  byte log: {byte_log[qid]}")
    
    print(f"Query ratio log: {query_ratio_dict}")
    
    if args.mode=='debug':
        print(f"Join cost distributions: {join_cost_dict}")
        if settings.benchmark not in dubug_logger.join_cost_distributions:
            dubug_logger.join_cost_distributions[settings.benchmark]={}
        dubug_logger.join_cost_distributions[settings.benchmark][tree_type] = join_cost_dict
        
        
        if settings.benchmark not in dubug_logger.query_cost_distributions:
            dubug_logger.query_cost_distributions[settings.benchmark]={}
        dubug_logger.query_cost_distributions[settings.benchmark][tree_type] = {}
        for qid in cost_log.keys():
            # Find index of -1 in cost_log, join elements before it as string, join elements after it as string, separated by space
            if -1 in cost_log[qid]:
                idx = cost_log[qid].index(-1)
                dubug_logger.query_cost_distributions[settings.benchmark][tree_type][qid]=[round(query_ratio_dict[qid],4),' '.join([str(i) for i in cost_log[qid][:idx]]),''.join([str(i) for i in cost_log[qid][idx+1:]])]
        
    # 2. Summarize table-level results
    final_join_result = {}
    for table in join_cost_dict.keys():
        final_join_result[table] = {
            'shuffle times': shuffle_times.get(table, 0),
            'hyper times': hyper_times.get(table, 0),
            'join cost': join_cost_dict[table]
        }
    
    # 3. Calculate averages
    query_cost_dict['avg'] = sum(list(query_cost_dict.values())) / len(query_cost_dict.keys())
    query_ratio_dict['avg'] = sum(list(query_ratio_dict.values())) / len(query_ratio_dict.keys())
    
    table_num=len(final_join_result.keys())
    
    shuffle_avg = sum(list(shuffle_times.values())) / table_num if shuffle_times else 0
    hyper_avg = sum(list(hyper_times.values())) / table_num if hyper_times else 0
    cost_avg = sum(list(join_cost_dict.values())) / table_num if join_cost_dict else 0
    
    final_join_result['avg'] = {
        'shuffle times': shuffle_avg,
        'hyper times': hyper_avg,
        'join cost': cost_avg
    }
    
    # 4. Calculate optimization time
    opt_time = [
        tot_order_time,
        sum(hyper_group_time) / len(hyper_group_time) if len(hyper_group_time) > 0 else 0,
        sum(shuffle_group_time) / len(shuffle_group_time) if len(shuffle_group_time) > 0 else 0
    ]
    
    return final_join_result, query_cost_dict, query_ratio_dict, opt_time


def processing_join_workload_old2(hypergraph,group_type,join_strategy='traditional',tree_type='paw'):
    shuffle_times,hyper_times={},{}
    join_cost_dict={}
    cost_log={}
    byte_log={}
    solved_tables={}
    input_queries=context.paw_queries if tree_type=='paw' else context.mto_queries if tree_type=='mto' else context.pac_queries
    shuffle_group_time,hyper_group_time=[],[]
    tot_order_time=0
    for q_id,join_query in enumerate(input_queries):
        print('processing query:',q_id)
        if _should_skip_query(q_id):
            continue
        if join_query['vectors']=={}:
            print(f'empty query : {settings.benchmark} {q_id}')
            continue
        query_cost_log,query_bytes_log=list(),list()
        solved_tables[q_id]=set()
        if join_query['join_relations']:
            start1_time=time.time()
            if join_strategy=='traditional':
                join_path=TraditionalJoinOrder(join_query['join_relations'],context.metadata).paths
            else:
                scan_block_dict={'card':{},'relation':[]}
                for join_op in join_query['join_relations']:
                    left_table,left_col_idx=list(join_op.items())[0]
                    right_table,right_col_idx=list(join_op.items())[1]
                    left_col,right_col=context.metadata[left_table]["used_cols"][left_col_idx],context.metadata[right_table]["used_cols"][right_col_idx]
                    left_query=join_query['vectors'][left_table]
                    right_query=join_query['vectors'][right_table]
                    is_hyper=True
                    if tree_type=='paw':
                        if f'{left_table}.{left_col}' not in scan_block_dict["card"]:
                            scan_block_dict["card"][f'{left_table}.{left_col}']=len(context.qdTreer.candidate_trees[left_table].query_single(left_query))
                        
                        if f'{right_table}.{right_col}' not in scan_block_dict["card"]:
                            scan_block_dict["card"][f'{right_table}.{right_col}']=len(context.qdTreer.candidate_trees[right_table].query_single(right_query))
                        is_hyper=False
                    else:
                        if left_col in hypergraph.hyper_nodes[left_table]:
                            tree=context.joinTreer.candidate_trees[left_table][left_col]
                        else:
                            tree=list(context.joinTreer.candidate_trees[left_table].values())[0]
                            is_hyper=False
                        if f'{left_table}.{left_col}' not in scan_block_dict["card"]:
                            scan_block_dict["card"][f'{left_table}.{left_col}']=len(tree.query_single(left_query))
                        
                        if right_col in hypergraph.hyper_nodes[right_table]:
                            tree=context.joinTreer.candidate_trees[right_table][right_col]
                        else:
                            tree=list(context.joinTreer.candidate_trees[right_table].values())[0]
                            is_hyper=False
                        if f'{right_table}.{right_col}' not in scan_block_dict["card"]:
                            scan_block_dict["card"][f'{right_table}.{right_col}']=len(tree.query_single(right_query))
                    
                    scan_block_dict["relation"].append([f'{left_table}.{left_col}',f'{right_table}.{right_col}','Hyper' if is_hyper else 'Shuffle'])
                jg=JoinGraph(scan_block_dict)
                join_path=jg.generate_MST()
            tot_order_time+=time.time()-start1_time   # 计算join reorder的时间
            temp_table_dfs=[]
            temp_joined_bytes=[]
            temp_shuffle_ops=[]
            
            for order_id,item in enumerate(join_path):
                join_queryset,joined_cols,joined_trees,joined_tables=[],[],[],[]
                is_hyper=True if item[2]==1 else False
                
                is_real_hyper=True
                join_ops=[(item[0].table,item[0].adj_col[item[1]]),(item[1].table,item[1].adj_col[item[0]])]
                
                for join_table,join_col in join_ops:
                    join_queryset.append(join_query['vectors'][join_table])
                    join_col_idx=context.metadata[join_table]["used_cols"].index(join_col)
                    joined_cols.append(join_col_idx)
                    if tree_type=='paw':
                        tree=context.qdTreer.candidate_trees[join_table]
                        is_real_hyper=False
                    else:
                        if join_col in hypergraph.hyper_nodes[join_table]:
                            tree=context.joinTreer.candidate_trees[join_table][join_col]
                        else:
                            tree=list(context.joinTreer.candidate_trees[join_table].values())[0]
                            is_real_hyper=False
                    joined_trees.append(tree)
                    joined_tables.append(join_table)
                
                if not is_hyper:
                    tree=None
                    cur_idx=0 if order_id==0 else 1
                    if order_id==len(join_path)-1 or join_path[order_id+1][2]==-1:
                        tree=joined_trees[cur_idx]
                        cur_table=joined_tables[cur_idx]
                        solved_tables[q_id].add(cur_table)
                        b_ids=tree.query_single(join_queryset[cur_idx])
                        table_dataset=[]
                        for b_id in b_ids:
                            table_dataset+=list(tree.nid_node_dict[b_id].dataset)
                        if hypergraph.benchmark=='imdb':
                            used_columns=[table_suffix[hypergraph.benchmark][cur_table]+"_"+col for col in tree.used_columns]
                        else:
                            used_columns=tree.used_columns
                        temp_table_dfs.append(pd.DataFrame(table_dataset,columns=used_columns))
                    temp_shuffle_ops.append((item[0].table,item[0].adj_col[item[1]],item[1].table,item[1].adj_col[item[0]]))
                    temp_joined_bytes.append(context.table_metadata[item[1].table]['read_line'])
                    continue
                else:
                    if len(joined_trees[0].query_single(join_queryset[0]))>len(joined_trees[1].query_single(join_queryset[1])):
                        join_queryset[0],join_queryset[1]=join_queryset[1],join_queryset[0]
                        joined_cols[0],joined_cols[1]=joined_cols[1],joined_cols[0]
                        joined_trees[0],joined_trees[1]=joined_trees[1],joined_trees[0]
                        joined_tables[0],joined_tables[1]=joined_tables[1],joined_tables[0]
                    left_table,right_table=joined_tables[0],joined_tables[1]

                    if is_real_hyper:
                        hyper_times[left_table]=hyper_times.get(left_table,0)+1
                        hyper_times[right_table]=hyper_times.get(left_table,0)+1
                    else:
                        shuffle_times[left_table]=shuffle_times.get(left_table,0)+1
                        shuffle_times[right_table]=shuffle_times.get(left_table,0)+1
                    
                    solved_tables[q_id].add(left_table)
                    solved_tables[q_id].add(right_table)
                    
                    join_eval = JoinEvaluator(join_queryset,joined_cols,joined_trees,joined_tables,settings.block_size,context.table_metadata,benchmark=hypergraph.benchmark)
                    hyper_shuffle_cost,hyper_shuffle_read_bytes, temp_joined_df, group_time=join_eval.compute_total_shuffle_hyper_cost(group_type,is_real_hyper)
                    join_cost_dict[left_table]=join_cost_dict.get(left_table,0)+hyper_shuffle_cost//2
                    join_cost_dict[right_table]=join_cost_dict.get(right_table,0)+hyper_shuffle_cost//2
                    if is_real_hyper:
                        hyper_group_time.append(group_time)
                    else:
                        shuffle_group_time.append(group_time)
                    query_cost_log.append(hyper_shuffle_cost)
                    query_bytes_log.append(hyper_shuffle_read_bytes)
                    temp_table_dfs.append(temp_joined_df)
                    temp_joined_bytes.append(context.table_metadata[left_table]['read_line']+context.table_metadata[right_table]['read_line'])
            
            if len(temp_table_dfs)>1:
                query_cost_log.append(-1)
                shuffle_cost=0
                shuffle_bytes=0
                last_temp_dfs=temp_table_dfs[0]
                last_temp_line=temp_joined_bytes[0]
                def pandas_hash_join(df_A, join_col_A, df_B, join_col_B):
                    if join_col_B in df_A.columns:
                        return df_A
                    if df_A.shape[0] > df_B.shape[0]:
                        df_A, df_B = df_B, df_A
                        join_col_A, join_col_B = join_col_B, join_col_A
                    df_B.drop_duplicates(subset=[join_col_B], keep='first')
                    merged_df = df_A.merge(df_B, how='inner', left_on=join_col_A, right_on=join_col_B)
                    return merged_df
                
                for i in range(1,len(temp_table_dfs)):
                    shuffle_cost+=(temp_table_dfs[i].shape[0]*3+last_temp_dfs.shape[0])
                    shuffle_bytes+=(temp_table_dfs[i].shape[0]*3*last_temp_line+last_temp_dfs.shape[0]*temp_joined_bytes[i])
                    join_table1,join_col1,join_table2,join_col2=temp_shuffle_ops[i-1]
                    if hypergraph.benchmark=='imdb':
                        join_col1=table_suffix[hypergraph.benchmark][join_table1]+"_"+join_col1
                        join_col2=table_suffix[hypergraph.benchmark][join_table2]+"_"+join_col2
                    last_temp_dfs=pandas_hash_join(last_temp_dfs,join_col1,temp_table_dfs[i],join_col2)
                    last_temp_line=temp_joined_bytes[i]+last_temp_line

                    shuffle_times[join_table1]=shuffle_times.get(join_table1,0)+1
                    shuffle_times[join_table2]=shuffle_times.get(join_table2,0)+1
                    join_cost_dict[join_table1]=join_cost_dict.get(join_table1,0)+shuffle_cost//2
                    join_cost_dict[join_table2]=join_cost_dict.get(join_table2,0)+shuffle_cost//2
                query_cost_log.append(shuffle_cost)
                query_bytes_log.append(shuffle_bytes)
        else:
            print('no join operation')
            for table, query_vector in join_query['vectors'].items():
                solved_tables[q_id].add(table)
                if tree_type=='paw':
                    tree=context.qdTreer.candidate_trees[table]
                else:
                    tree=list(context.joinTreer.candidate_trees[table].values())[0]
                b_ids=tree.query_single(query_vector)
                scan_cost=sum([tree.nid_node_dict[b_id].node_size for b_id in b_ids])
                
                join_cost_dict[table]=join_cost_dict.get(table,0)+scan_cost
                query_cost_log.append(scan_cost)
                query_bytes_log.append(context.table_metadata[table]['read_line']*scan_cost)
        cost_log[q_id]=query_cost_log
        byte_log[q_id]=query_bytes_log
    
    query_cost_dict={}
    query_ratio_dict={}
    for qid in cost_log.keys():
        query_cost_dict[qid]=sum(list(cost_log[qid]))
        query_ratio_dict[qid]=sum(list(byte_log[qid]))/sum([sum(context.table_metadata[table]['width'].values())*context.table_metadata[table]['rows'] for table in solved_tables[qid]])
        print(f"Query {qid} cost log: {cost_log[qid]}  |  ratio log: {byte_log[qid]}")    
    print(f"Query ratio log: {query_ratio_dict}")    
    final_join_result={}
    for table in join_cost_dict.keys():
        final_join_result[table]={'shuffle times':shuffle_times.get(table,0),'hyper times':hyper_times.get(table,0),'join cost':join_cost_dict[table]}
        
    query_cost_dict['avg']=sum(list(query_cost_dict.values()))/len(query_cost_dict.keys())
    query_ratio_dict['avg']=sum(list(query_ratio_dict.values()))/len(query_ratio_dict.keys())
    final_join_result['avg']={'shuffle times':sum(list(shuffle_times.values()))/len(shuffle_times.keys()) if shuffle_times else 0,'hyper times':sum(list(hyper_times.values()))/len(hyper_times.keys()) if hyper_times else 0,'join cost':sum(list(join_cost_dict.values()))/len(join_cost_dict.keys()) if join_cost_dict else 0}
    
    opt_time=[tot_order_time,sum(hyper_group_time)/len(hyper_group_time) if len(hyper_group_time)>0 else 0,sum(shuffle_group_time)/len(shuffle_group_time) if len(shuffle_group_time)>0 else 0]
    return final_join_result,query_cost_dict,query_ratio_dict,opt_time


def select_columns_by_PAW(group_type=0, strategy='traditional'):
    # Build hypergraph structure
    hypergraph=None
    if settings.benchmark=='tpch':
        hypergraph = HypergraphTPC() 
    elif settings.benchmark=='imdb':
        hypergraph = JobHypergraph()
    elif settings.benchmark=='tpcds':
        hypergraph = HypergraphTPCDS()
    total_opt_time=[]
    tree_opt_time=0
    for table in hypergraph.hyper_nodes:
        tree_opt_time+=context.qdTreer.candidate_trees[table].build_time
    total_opt_time.append(tree_opt_time)
    final_result, query_costs, query_ratio, group_opt_time = processing_join_workload(hypergraph, group_type, strategy, tree_type='paw')
    # total_opt_time+=group_opt_time
    print("Final Column Selection Cost of PAW is: ",sum(list(query_costs.values()))/len(query_costs.keys()))
    print("Average Data Cost Ratio of PAW is: ",sum(list(query_ratio.values()))/len(query_ratio.keys()))
    print("Total Opt Time of PAW is: ",total_opt_time)
    hypergraph.show_selections()
    hypergraph_save_path=f'{base_dir}/../layouts/bs={settings.block_size}/{hypergraph.benchmark}/join_key_selection/paw-hgraph.pkl'
    os.makedirs(os.path.dirname(hypergraph_save_path), exist_ok=True)
    
    with open(hypergraph_save_path,'wb') as f:
        pickle.dump(hypergraph,f)
    return final_result, query_costs,query_ratio, sum(total_opt_time)
    

def select_columns_by_AD_MTO(group_type=0, strategy='traditional'):
    hypergraph=None
    if settings.benchmark=='tpch':
        hypergraph = HypergraphTPC() 
    elif settings.benchmark=='imdb':
        hypergraph = JobHypergraph()
    elif settings.benchmark=='tpcds':
        hypergraph = HypergraphTPCDS()
    total_opt_time=[]
    tree_opt_time=0
    for table in hypergraph.hyper_nodes:
        col=context.metadata[table]['used_cols'][0]
        if col in hypergraph.candidate_nodes[table]:
            hypergraph.hyper_nodes[table].append(col)
            tree_opt_time+=context.joinTreer.candidate_trees[table][col].build_time
    total_opt_time.append(tree_opt_time)
    final_result, query_costs, query_ratio, group_opt_time = processing_join_workload(hypergraph, group_type, strategy, tree_type='mto')
    # total_opt_time+=group_opt_time
    print("Final Column Selection Cost of AD-MTO is: ",sum(list(query_costs.values()))/len(query_costs.keys()))
    print("Average Data Cost Ratio of AD-MTO is: ",sum(list(query_ratio.values()))/len(query_ratio.keys()))
    print("Total Opt Time of AD-MTO is: ",total_opt_time)
    hypergraph.show_selections()
    hypergraph_save_path=f'{base_dir}/../layouts/bs={settings.block_size}/{hypergraph.benchmark}/join_key_selection/ad-mto-hgraph.pkl'
    os.makedirs(os.path.dirname(hypergraph_save_path), exist_ok=True)
    with open(hypergraph_save_path,'wb') as f:
        pickle.dump(hypergraph,f)

    return final_result, query_costs, query_ratio, sum(total_opt_time)

def select_columns_by_JT(group_type,strategy='traditional',disabled_joinkey_selection=False):

    adj_matrix = {}
    for query in context.pac_queries:
        for join_op in query["join_relations"]:
            join_relations=set()
            for table,col in join_op.items():
                join_relations.add(f'{table}.{context.metadata[table]["used_cols"][col]}')
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
    elif settings.benchmark=='tpcds':
        hypergraph = HypergraphTPCDS()
    total_opt_time=[]


    tree_opt_time=0
    if disabled_joinkey_selection:
        hypergraph.clear_hyper_nodes()
        for table in hypergraph.hyper_nodes:
            col=context.metadata[table]['used_cols'][0]
            if col in hypergraph.candidate_nodes[table]:
                hypergraph.hyper_nodes[table].append(col)
                tree_opt_time+=context.joinTreer.candidate_trees[table][col].build_time       
    else:
        for table in hypergraph.candidate_nodes:
            col_freq_dict={}
            for source_col in hypergraph.candidate_nodes[table]:
                if f'{table}.{source_col}' in adj_matrix:
                    for key,freq in adj_matrix[f'{table}.{source_col}'].items():
                        targe_table=key.split('.')[0]
                        tot_freq=freq*context.table_metadata[targe_table]['rows']
                else:
                    tot_freq=0
                col_freq_dict[source_col]=tot_freq
        
            col_list=sorted(col_freq_dict.items(), key=lambda x: x[1], reverse=True)
        
            for i in range(1,settings.replication_factor+1):
                if i > len(col_list):
                    break
                hypergraph.hyper_nodes[table].append(col_list[i-1][0])

        for table in hypergraph.hyper_nodes:
            for col in hypergraph.hyper_nodes[table]:
                tree_opt_time+=context.joinTreer.candidate_trees[table][col].build_time
    
    total_opt_time.append(tree_opt_time)
    final_result, query_costs, query_ratio, group_opt_time = processing_join_workload(hypergraph, group_type, strategy, tree_type='pac')
    # total_opt_time+=group_opt_time
    print("Final Column Selection Cost of PAC-Tree is: ",sum(list(query_costs.values()))/len(query_costs.keys()))
    print("Average Data Cost Ratio of PAC-Tree is: ",sum(list(query_ratio.values()))/len(query_ratio.keys()))
    print("Total Opt Time (tree,order,group) of PAC-Tree is: ",total_opt_time)
    hypergraph.show_selections()
    hypergraph_save_path=f'{base_dir}/../layouts/bs={settings.block_size}/{hypergraph.benchmark}/join_key_selection/jt-hgraph.pkl'
    os.makedirs(os.path.dirname(hypergraph_save_path), exist_ok=True)
    with open(hypergraph_save_path,'wb') as f:
        pickle.dump(hypergraph,f)
    return final_result, query_costs, query_ratio, sum(total_opt_time)

def save_experiment_result(experiment_result,disabled_prim_reorder):
    import pandas as pd
    import os

    base_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = f"{base_dir}/../experiment/result/sf={settings.scale_factor}/bs={settings.block_size}/rep={settings.replication_factor}/"
    os.makedirs(result_dir, exist_ok=True)
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
            "ADP": "AD-MTO_Scan Block",
            "JT": "PAC-Tree_Scan Block",
        },
        inplace=True
    ) 
    os.makedirs(result_dir, exist_ok=True)
    res_saved_path=f"{result_dir}/{settings.benchmark}_results.xlsx"
    with pd.ExcelWriter(res_saved_path) as writer:
        merged_query_df.to_excel(writer, sheet_name="Query_Cost", index=False)
        merged_df.to_excel(writer, sheet_name="Table_Cost", index=False)
        df_time.to_excel(writer, sheet_name="Time_Cost", index=False)


def pretty_print_dubgger_query_join_info():
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    
    console = Console()
    benchmarks = list(dubug_logger.query_cost_distributions.keys())
    for benchmark in benchmarks:
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("QueryID", style="bold magenta")
        methods = sorted(list(dubug_logger.query_cost_distributions[benchmark].keys()))
        for method in methods:
            table.add_column(f"{method} Ratio", justify="right", style=get_method_style(method))
            table.add_column(f"{method} Hyper", justify="right", style=get_method_style(method))
            table.add_column(f"{method} Shuffle", justify="right", style=get_method_style(method))
        all_qids = set()
        for method in methods:
            all_qids.update(dubug_logger.query_cost_distributions[benchmark][method].keys())
        for qid in sorted(list(all_qids)):
            row = [str(qid)]
            for method in methods:
                if qid in dubug_logger.query_cost_distributions[benchmark][method]:
                    scan_ratio,hyper_cost, shuffle_cost = dubug_logger.query_cost_distributions[benchmark][method][qid]
                    row.append(str(scan_ratio))
                    row.append(hyper_cost)
                    row.append(shuffle_cost)
                else:
                    row.append("N/A")
                    row.append("N/A")
                    row.append("N/A")
            table.add_row(*row)
        panel = Panel(table, 
                     title=f"[bold yellow]Query Cost Distributions for {benchmark}[/bold yellow]", 
                     border_style="blue", 
                     expand=False)
        console.print(panel)
        console.print("\n")

def get_method_style(method):
    """Return the color style for a given method."""
    styles = {
        "paw": "green",
        "mto": "blue",
        "pac": "yellow",
        "adp": "cyan"
    }
    return styles.get(method.lower(), "white")


def pretty_print_dubgger_table_join_info():
    from rich.console import Console
    from rich.table import Table
    console = Console()
    table = Table(show_header=True, header_style="bold cyan")
    benchmarks = list(dubug_logger.join_cost_distributions.keys())
    table.add_column("Benchmark", style="bold magenta")
    table.add_column("Table", style="dim")
    all_methods = set()
    for benchmark in benchmarks:
        all_methods.update(dubug_logger.join_cost_distributions[benchmark].keys())
    method_list = sorted(list(all_methods))  # Sort to ensure consistent order
    for method in method_list:
        table.add_column(method, justify="right", style=get_method_style(method))
    for benchmark in benchmarks:
        all_tables = set()
        benchmark_methods = dubug_logger.join_cost_distributions[benchmark].keys()
        for method in benchmark_methods:
            all_tables.update(dubug_logger.join_cost_distributions[benchmark][method].keys())
        primary_method = next(iter(benchmark_methods))
        sorted_tables = sorted(
            [t for t in all_tables if t != "avg"],
            key=lambda t: dubug_logger.join_cost_distributions[benchmark][primary_method].get(t, 0),
            reverse=True
        )
        if "avg" in all_tables:
            sorted_tables.append("avg")
        for i, table_name in enumerate(sorted_tables):
            row = []
            if i == 0:
                row.append(benchmark)
            else:
                row.append("")
            row.append(table_name)
            for method in method_list:
                if method in benchmark_methods:
                    cost = dubug_logger.join_cost_distributions[benchmark][method].get(table_name, "N/A")
                    row.append(f"{cost:,}" if isinstance(cost, (int, float)) else cost)
                else:
                    row.append("N/A")
            if table_name == "avg":
                table.add_row(*row, style="bold")
            else:
                table.add_row(*row)
    console.print("\n[bold yellow]Join Cost Distributions By Method[/bold yellow]\n")
    console.print(table)

def get_method_style(method):
    styles = {
        "paw": "green",
        "mto": "blue",
        "pac": "yellow",
        "adp": "cyan"
    }
    return styles.get(method.lower(), "white")


def run_evaluation(save_result=False, disabled_prim_reorder=False):
    """
    disabled_prim_reorder: 是否使用 基于prim算法的join reorder。若为True，则不使用join reorder
    """
    load_tree_context()
    
    experiment_data = {}
    # logging.basicConfig(filename=f'{base_dir}/../logs/column_selection_experiment.log',
    #                     level=logging.INFO,
    #                     format='%(asctime)s - %(levelname)s - %(message)s')

    using_join_order_strategy='traditional' if disabled_prim_reorder else 'prim'
    def run_PAW():
        res_PAW, cost_PAW, ratio_PAW, total_opt_time = select_columns_by_PAW(strategy='traditional', group_type=0) 
        return [res_PAW, cost_PAW, ratio_PAW, total_opt_time]
    
    def run_AD_MTO():
        res_ADP, cost_ADP, ratio_ADP, total_opt_time = select_columns_by_AD_MTO(strategy='traditional', group_type=0)
        return [res_ADP, cost_ADP, ratio_ADP, total_opt_time]
        
    def run_JT():
        res_JT, cost_JT,ratio_JT, total_opt_time = select_columns_by_JT(group_type=1, strategy=using_join_order_strategy, disabled_joinkey_selection=False) 
        return [res_JT, cost_JT, ratio_JT, total_opt_time]
    
    experiment_data["PAW"]=run_PAW()
    experiment_data["ADP"]=run_AD_MTO()
    experiment_data["JT"]=run_JT()
    
    if args.mode=='debug':
        pretty_print_dubgger_table_join_info()
        pretty_print_dubgger_query_join_info()
    
    # exit(-1)
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
    for item in [1000, 5000, 20000, 50000]:
        settings.block_size=item
        run_evaluation(save_result=True, disabled_prim_reorder=True)


def scaling_replication():
    for item in [1,2,3,4]:
        settings.replication_factor=item
        run_evaluation(save_result=True, disabled_prim_reorder=False)
        

def test_model_overhead_by_scaling_scale():
    for sf in [5,10,20,50]:
        settings.scale_factor=sf
        # settings.block_size=int(10000/sf)
        run_evaluation(save_result=True, disabled_prim_reorder=False)


if args.command == 0:  #basic experiments
    settings.replication_factor=3 #defult
    run_evaluation(save_result=True, disabled_prim_reorder=False)
elif args.command == 1: # sensitivity analysis
    scaling_block_size()
elif args.command == 2: 
    scaling_replication()
elif args.command == 3: 
    test_model_overhead_by_scaling_scale()
else:
    print("Invalid command")
