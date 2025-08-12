"""
请为我根据TPC-H schema信息，创建一个超图结构。

每个超节点对应一个表，每个超节点中存在多个节点（通常是外键和主键，请先给出所有候选列）。至于超图的边，这是不重要的，它将由具体的查询来决定。

现在你的任务是根据所有已有的join查询，决定超图中每个超节点里面的列，即选择每个超节点中哪些列。超节点的内容表示该表数据将根据每个单个节点列创建单独的分区方案，
超节点中的多个节点列则对应存在多个分区副本，这里允许对该表进行复制，作为副本。根据你选择的列使整体join查询的执行开销最低。

注意：请用python语言实现这一功能。
你可以给出几个示例join查询，以及你的超图结构，以及你的选择结果。具体选择超节点中列的算法，可以先不写，后续可讨论。join开销的计算也可以先不写，封装到一个函数中，后续讨论。
"""

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
from rich.console import Console
from rich.logging import RichHandler
import logging

base_dir=os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--command",type=int, default=0, help="choose the command to run")
parser.add_argument("--benchmark",type=str, default='tpcds', help="choose the evaluated benchmark")
parser.add_argument("--mode", type=str, default='debug', help="choose the mode to run")
parser.add_argument("--init", type=bool, default=False, help="initialize base layouts and join queries")


# python join_key_selector.py --init --benchmark=tpch
# python join_key_selector.py --command=0 --benchmark=tpch

args = parser.parse_args()
settings=JoinSelectorSettings()
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
        # candidate_nodes = {
        #     "call_center": ["cc_call_center_sk"],
        #     "catalog_page": ["cp_catalog_page_sk"],
        #     "catalog_returns": ["cr_returned_date_sk",  "cr_returned_time_sk","cr_item_sk","cr_refunded_customer_sk","cr_returning_customer_sk"],
        #     "customer_address": ["ca_address_sk"],
        #     "customer_demographics": ["cd_demo_sk"],
        #     "customer": ["c_customer_sk", "c_current_cdemo_sk", "c_current_hdemo_sk", "c_current_addr_sk","c_first_shipto_date_sk","c_first_sales_date_sk"],
        #     "date_dim": ["d_date_sk"],
        #     "dbgen_version": ["dv_version"],
        #     "household_demographics": ["hd_demo_sk","hd_income_band_sk"],
        #     "income_band": ["ib_income_band_sk"],
        #     "inventory": ["inv_date_sk", "inv_item_sk", "inv_warehouse_sk"],
        #     "item": ["i_item_sk"],
        #     "promotion": ["p_promo_sk", "p_item_sk"],
        #     "reason": ["r_reason_sk"],
        #     "ship_mode": ["sm_ship_mode_sk"],
        #     "store_sales": ["ss_sold_date_sk", "ss_item_sk", "ss_customer_sk", "ss_cdemo_sk", "ss_hdemo_sk", "ss_addr_sk"],
        #     "store": ["s_store_sk"],
        #     "time_dim": ["t_time_sk"],
        #     "warehouse": ["w_warehouse_sk"],
        #     "web_page": ["wp_web_page_sk"],
        #     "web_returns": ["wr_returned_date_sk", "wr_item_sk", "wr_order_number", "wr_returning_customer_sk", "wr_returned_time_sk"],
        #     "web_sales": ["ws_sold_date_sk"],
        #     "web_site": ["web_site_sk"],
        # }
        candidate_nodes = {
            # 以下表在查询中有连接
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
            # 以下表在当前查询集中未出现连接，为其选择最可能的连接键
            "call_center": ["cc_call_center_sk"],  # 主键
            "catalog_page": ["cp_catalog_page_sk"],  # 主键
            "catalog_returns": ["cr_item_sk", "cr_order_number", "cr_returned_date_sk"],  # 复合主键和常用外键
            "income_band": ["ib_income_band_sk"],  # 主键
            "reason": ["r_reason_sk"],  # 主键
            "ship_mode": ["sm_ship_mode_sk"],  # 主键
            "time_dim": ["t_time_sk"],  # 主键
            "warehouse": ["w_warehouse_sk"],  # 主键
            "web_page": ["wp_web_page_sk"],  # 主键
            "web_returns": ["wr_item_sk", "wr_order_number", "wr_returned_date_sk"],  # 复合主键和常用外键
            "web_site": ["web_site_sk"]  # 主键
        }
        super().__init__('tpcds', candidate_nodes)

# 为所有表创建候选join trees
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
    # 配置 rich 控制台处理程序
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

# 直接读取 joinTreer, join_queries等关键信息
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
    # 保存 join-trees.pkl 文件
    with open(f'{join_trees_dir}/join-trees.pkl', 'wb') as f:
        pickle.dump(joinTreer, f)

    # # Create QD trees for all tables
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
      
    # 定义保存查询的基础目录
    queries_base_dir = f'{base_dir}/../layouts/bs={settings.block_size}/{hypergraph.benchmark}/used_queries'
    # 创建多级目录，如果目录已存在则不会报错
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

    # ~~~~~~~~读取表的部分元数据信息~~~~~~~~~
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
     # 查询测试日志： tpch:删除q_id=15, imdb:删除q_id=7
    return (settings.benchmark == 'imdb' and q_id == 7) or \
           (settings.benchmark == 'tpch' and q_id == 6) or \
           (settings.benchmark == 'tpcds' and q_id == -1)
    # if q_id not in [10,11,12,13,14,15,16,17]:
        # if q_id not in [17]:
        #     continue


def processing_join_workload(hypergraph, group_type, join_strategy='traditional', tree_type='paw'):
    """
    处理 join 工作负载，计算当前超图中 hyper 边和 shuffle 边的位置及总成本
    
    Args:
        hypergraph: 超图结构
        group_type: 分组类型（0 或 1）
        join_strategy: join 策略（'traditional' 或 'prim'）
        tree_type: 树类型（'paw', 'mto' 或 'pac'）
        
    Returns:
        final_join_result: 表级别的 join 结果统计
        query_cost_dict: 查询级别的成本统计
        query_ratio_dict: 查询级别的比率统计
        opt_time: 优化时间统计
    """
    # 初始化统计跟踪器
    shuffle_times, hyper_times = {}, {}
    join_cost_dict = {}
    cost_log, byte_log = {}, {}
    solved_tables = {}
    shuffle_group_time, hyper_group_time = [], []
    tot_order_time = 0
    
    # 选择适当的查询集
    input_queries = context.paw_queries if tree_type == 'paw' else \
                  context.mto_queries if tree_type == 'mto' else context.pac_queries
    
    # ========== 处理每个查询 ==========
    for q_id, join_query in enumerate(input_queries):
        print('processing query:', q_id)

        # 跳过特殊查询
        if _should_skip_query(q_id) or join_query['vectors'] == {}:
            if join_query['vectors'] == {}:
                print(f'empty query : {settings.benchmark} {q_id}')
            continue

        # if q_id!=2:
        #     continue
        
        query_cost_log, query_bytes_log = [], []
        solved_tables[q_id] = set()
        
        # ---------- 无 join 关系的查询处理 ----------
        if not join_query['join_relations']:
            print('no join operation')
            for table, query_vector in join_query['vectors'].items():
                solved_tables[q_id].add(table)
                
                # 选择合适的树
                if tree_type == 'paw':
                    tree = context.qdTreer.candidate_trees[table]
                else:
                    tree = list(context.joinTreer.candidate_trees[table].values())[0]
                
                # 计算扫描成本
                b_ids = tree.query_single(query_vector)
                scan_cost = sum([tree.nid_node_dict[b_id].node_size for b_id in b_ids])
                join_cost_dict[table] = join_cost_dict.get(table, 0) + scan_cost
                query_cost_log.append(scan_cost)
                query_bytes_log.append(context.table_metadata[table]['read_line'] * scan_cost)
                logger.info("Just scan table: %s, cost: %s", table, scan_cost)
                
            cost_log[q_id] = query_cost_log
            byte_log[q_id] = query_bytes_log
            continue
        
        # ---------- 有 join 关系的查询处理 ----------
        # 1. 确定 join 顺序
        start1_time = time.time()
        join_path = []
        
        if join_strategy == 'traditional':
            join_path = TraditionalJoinOrder(join_query['join_relations'], context.metadata).paths
        else:
            # 构建用于 Prim 算法的扫描块字典
            scan_block_dict = {'card': {}, 'relation': []}
            
            for join_op in join_query['join_relations']:
                left_table, left_col_idx = list(join_op.items())[0]
                right_table, right_col_idx = list(join_op.items())[1]
                left_col = context.metadata[left_table]["used_cols"][left_col_idx]
                right_col = context.metadata[right_table]["used_cols"][right_col_idx]
                left_query = join_query['vectors'][left_table]
                right_query = join_query['vectors'][right_table]
                is_hyper = True
                
                # PAW 树类型的处理
                if tree_type == 'paw':
                    if f'{left_table}.{left_col}' not in scan_block_dict["card"]:
                        scan_block_dict["card"][f'{left_table}.{left_col}'] = len(context.qdTreer.candidate_trees[left_table].query_single(left_query))
                    
                    if f'{right_table}.{right_col}' not in scan_block_dict["card"]:
                        scan_block_dict["card"][f'{right_table}.{right_col}'] = len(context.qdTreer.candidate_trees[right_table].query_single(right_query))
                    is_hyper = False
                # MTO 或 PAC 树类型的处理
                else:
                    # 左表树处理
                    if left_col in hypergraph.hyper_nodes[left_table]:
                        tree = context.joinTreer.candidate_trees[left_table][left_col]
                    else:
                        tree = list(context.joinTreer.candidate_trees[left_table].values())[0]
                        is_hyper = False
                    
                    if f'{left_table}.{left_col}' not in scan_block_dict["card"]:
                        scan_block_dict["card"][f'{left_table}.{left_col}'] = len(tree.query_single(left_query))
                    
                    # 右表树处理
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
            
            # 使用 JoinGraph 生成 MST
            jg = JoinGraph(scan_block_dict,hypergraph.hyper_nodes)
            join_path = jg.generate_MST()
        
        tot_order_time += time.time() - start1_time  # 计算 join reorder 的时间
        
        # 2. 处理 join 路径
        temp_table_dfs = []
        temp_joined_bytes = []
        temp_shuffle_ops = []
        
        # 遍历 join 路径中的每一项
        for order_id, item in enumerate(join_path):
            join_queryset, joined_cols, joined_trees, joined_tables = [], [], [], []
            is_hyper = True if item[2] == 1 else False
            
            is_real_hyper = True
            join_ops = [
                (item[0].table, item[0].adj_col[item[1]]),
                (item[1].table, item[1].adj_col[item[0]])
            ]
            
            # 收集 join 操作的相关信息
            for join_table, join_col in join_ops:
                join_queryset.append(join_query['vectors'][join_table])
                join_col_idx = context.metadata[join_table]["used_cols"].index(join_col)
                joined_cols.append(join_col_idx)
                
                # 选择合适的树
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
            
            # 处理 shuffle join
            if not is_hyper:
                cur_idx = 0 if order_id == 0 else 1
                
                # 当前是最后一条路径或下一条不是 hyper
                if order_id == len(join_path) - 1 or join_path[order_id+1][2] == -1:
                    tree = joined_trees[cur_idx]
                    cur_table = joined_tables[cur_idx]
                    solved_tables[q_id].add(cur_table)
                    
                    # 获取表数据
                    b_ids = tree.query_single(join_queryset[cur_idx])
                    table_dataset = []
                    for b_id in b_ids:
                        table_dataset += list(tree.nid_node_dict[b_id].dataset)
                    
                    # 设置列名
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
            
            # 处理 hyper join
            # 确保较小的表在左边（优化）
            if len(joined_trees[0].query_single(join_queryset[0])) > len(joined_trees[1].query_single(join_queryset[1])):
                join_queryset[0], join_queryset[1] = join_queryset[1], join_queryset[0]
                joined_cols[0], joined_cols[1] = joined_cols[1], joined_cols[0]
                joined_trees[0], joined_trees[1] = joined_trees[1], joined_trees[0]
                joined_tables[0], joined_tables[1] = joined_tables[1], joined_tables[0]
            
            left_table, right_table = joined_tables[0], joined_tables[1]
            
            # 更新统计
            if is_real_hyper:
                hyper_times[left_table] = hyper_times.get(left_table, 0) + 1
                hyper_times[right_table] = hyper_times.get(right_table, 0) + 1
            else:
                shuffle_times[left_table] = shuffle_times.get(left_table, 0) + 1
                shuffle_times[right_table] = shuffle_times.get(right_table, 0) + 1
            
            solved_tables[q_id].add(left_table)
            solved_tables[q_id].add(right_table)
            
            # 执行 join 评估
            join_eval = JoinEvaluator(
                join_queryset, joined_cols, joined_trees, joined_tables,
                settings.block_size, context.table_metadata, benchmark=hypergraph.benchmark
            )
            hyper_shuffle_cost, hyper_shuffle_read_bytes, temp_joined_df, group_time = join_eval.compute_total_shuffle_hyper_cost(group_type, is_real_hyper)
            
            # 更新统计
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
        
        # 3. 处理多表 join
        if len(temp_table_dfs) > 1:
            query_cost_log.append(-1)  # 标记多表 join 开始
            shuffle_cost = 0
            shuffle_bytes = 0
            last_temp_dfs = temp_table_dfs[0]
            last_temp_line = temp_joined_bytes[0]
            
            # 定义 hash join 函数
            def pandas_hash_join(df_A, join_col_A, df_B, join_col_B):
                if join_col_B in df_A.columns:
                    return df_A
                if df_A.shape[0] > df_B.shape[0]:
                    df_A, df_B = df_B, df_A
                    join_col_A, join_col_B = join_col_B, join_col_A
                df_B = df_B.drop_duplicates(subset=[join_col_B], keep='first')
                merged_df = df_A.merge(df_B, how='inner', left_on=join_col_A, right_on=join_col_B)
                return merged_df
            
            # 执行多表 join
            for i in range(1, len(temp_table_dfs)):
                
                join_table1, join_col1, join_table2, join_col2 = temp_shuffle_ops[i-1]
                
                sample_rate1,sample_rate2 = 1, 1

                if join_col1 in context.joinTreer.candidate_trees[join_table1] and hasattr(context.joinTreer.candidate_trees[join_table1][join_col1], 'sample_rate'):
                    sample_rate1=context.joinTreer.candidate_trees[join_table1][join_col1].sample_rate

                if join_col2 in context.joinTreer.candidate_trees[join_table2] and hasattr(context.joinTreer.candidate_trees[join_table2][join_col2], 'sample_rate'):
                    sample_rate2=context.joinTreer.candidate_trees[join_table2][join_col2].sample_rate
                
                data_sample_rate=min(sample_rate1,sample_rate2)
                
                # 计算成本
                shuffle_cost += int((temp_table_dfs[i].shape[0] * 3 + last_temp_dfs.shape[0]* 1)/data_sample_rate)
                shuffle_bytes += int((temp_table_dfs[i].shape[0] * 3 * last_temp_line + 
                                 last_temp_dfs.shape[0]* 1 * temp_joined_bytes[i])/data_sample_rate)
                
                if hypergraph.benchmark == 'imdb':
                    join_col1 = table_suffix[hypergraph.benchmark][join_table1] + "_" + join_col1
                    join_col2 = table_suffix[hypergraph.benchmark][join_table2] + "_" + join_col2
                
                # 执行 join
                last_temp_dfs = pandas_hash_join(last_temp_dfs, join_col1, temp_table_dfs[i], join_col2)
                last_temp_line = temp_joined_bytes[i] + last_temp_line
                
                # 更新统计
                shuffle_times[join_table1] = shuffle_times.get(join_table1, 0) + 1
                shuffle_times[join_table2] = shuffle_times.get(join_table2, 0) + 1
                join_cost_dict[join_table1] = join_cost_dict.get(join_table1, 0) + shuffle_cost // 2
                join_cost_dict[join_table2] = join_cost_dict.get(join_table2, 0) + shuffle_cost // 2
                
                logger.info("Join %s and %s, shuffle cost: %s", join_table1, join_table2, shuffle_cost)
            
            query_cost_log.append(shuffle_cost)
            query_bytes_log.append(shuffle_bytes)
        
        cost_log[q_id] = query_cost_log
        byte_log[q_id] = query_bytes_log
    
    # ========== 汇总结果 ==========
    # 1. 计算查询成本和比率
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
            # 找到cost_log中-1的索引，将其前面的元素，转化为str连接在一起，后面的元素也转化为str连接在一起，中间用空格隔开
            if -1 in cost_log[qid]:
                idx = cost_log[qid].index(-1)
                dubug_logger.query_cost_distributions[settings.benchmark][tree_type][qid]=[round(query_ratio_dict[qid],4),' '.join([str(i) for i in cost_log[qid][:idx]]),''.join([str(i) for i in cost_log[qid][idx+1:]])]
        
    # 2. 汇总表级结果
    final_join_result = {}
    for table in join_cost_dict.keys():
        final_join_result[table] = {
            'shuffle times': shuffle_times.get(table, 0),
            'hyper times': hyper_times.get(table, 0),
            'join cost': join_cost_dict[table]
        }
    
    # 3. 计算平均值
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
    
    # 4. 计算优化时间
    opt_time = [
        tot_order_time,
        sum(hyper_group_time) / len(hyper_group_time) if len(hyper_group_time) > 0 else 0,
        sum(shuffle_group_time) / len(shuffle_group_time) if len(shuffle_group_time) > 0 else 0
    ]
    
    return final_join_result, query_cost_dict, query_ratio_dict, opt_time



# 现在，我已经给出了每个超节点初步的候选可以构建超图的边结构了。
# 定义函数：可以大致估算超图中，该表所在边的潜在join成本
def processing_join_workload_old2(hypergraph,group_type,join_strategy='traditional',tree_type='paw'):
    # 计算当前超图中hyper边和shuffle边的位置，以及估算的总成本
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
        # 计算hyper cost，便于确定join order
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
                            # 此时，无法在被选择树中无法直接找到对应的join_tree，
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
                # join_path=TraditionalJoinOrder(join_query['join_relations'],metadata).paths
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
                
                # shuffle join操作
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
                # hyper join 或 shuffle join 操作
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
            # 多表join操作
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
            "ADP": "AD-MTO_Scan Block",
            "JT": "PAC-Tree_Scan Block",
        },
        inplace=True
    ) 
    os.makedirs(result_dir, exist_ok=True)
    # res_saved_path=f"{result_dir}/{settings.benchmark}_results.xlsx" if disabled_prim_reorder else f"{result_dir}/{settings.benchmark}_results_enable_prim.xlsx"
    res_saved_path=f"{result_dir}/{settings.benchmark}_results.xlsx"
    with pd.ExcelWriter(res_saved_path) as writer:
        merged_query_df.to_excel(writer, sheet_name="Query_Cost", index=False)
        merged_df.to_excel(writer, sheet_name="Table_Cost", index=False)
        df_time.to_excel(writer, sheet_name="Time_Cost", index=False)


def pretty_print_dubgger_query_join_info():
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    
    # 创建控制台
    console = Console()
    
    # 获取所有基准测试
    benchmarks = list(dubug_logger.query_cost_distributions.keys())
    
    for benchmark in benchmarks:
        # 创建表格
        table = Table(show_header=True, header_style="bold cyan")
        
        # 添加基本列
        table.add_column("QueryID", style="bold magenta")
        
        # 确定所有可能的方法
        methods = sorted(list(dubug_logger.query_cost_distributions[benchmark].keys()))
        
        # 为每个方法添加两列（hyper和shuffle）
        for method in methods:
            table.add_column(f"{method} Ratio", justify="right", style=get_method_style(method))
            table.add_column(f"{method} Hyper", justify="right", style=get_method_style(method))
            table.add_column(f"{method} Shuffle", justify="right", style=get_method_style(method))
        
        # 获取所有查询ID
        all_qids = set()
        for method in methods:
            all_qids.update(dubug_logger.query_cost_distributions[benchmark][method].keys())
        
        # 为每个查询ID创建行
        for qid in sorted(list(all_qids)):
            row = [str(qid)]
            
            # 添加每个方法的成本
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
            
            # 添加行
            table.add_row(*row)
        
        # 创建一个带标题的面板包含表格
        panel = Panel(table, 
                     title=f"[bold yellow]Query Cost Distributions for {benchmark}[/bold yellow]", 
                     border_style="blue", 
                     expand=False)
        
        # 打印面板
        console.print(panel)
        console.print("\n")

def get_method_style(method):
    """返回每个方法的颜色样式"""
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
    
    # 创建控制台
    console = Console()
    
    # 创建主表格
    table = Table(show_header=True, header_style="bold cyan")
    
    # 获取所有基准测试和方法
    benchmarks = list(dubug_logger.join_cost_distributions.keys())
    # 添加基本列
    table.add_column("Benchmark", style="bold magenta")
    table.add_column("Table", style="dim")
    
    # 确定所有可能的方法
    all_methods = set()
    for benchmark in benchmarks:
        all_methods.update(dubug_logger.join_cost_distributions[benchmark].keys())
    
    method_list = sorted(list(all_methods))  # 排序以确保一致的顺序
    
    # 添加方法列
    for method in method_list:
        table.add_column(method, justify="right", style=get_method_style(method))
    
    # 为每个基准测试创建行
    for benchmark in benchmarks:
        # 获取这个基准测试中所有表的集合
        all_tables = set()
        benchmark_methods = dubug_logger.join_cost_distributions[benchmark].keys()
        
        for method in benchmark_methods:
            all_tables.update(dubug_logger.join_cost_distributions[benchmark][method].keys())
        
        # 按成本排序表（使用第一个可用方法）
        primary_method = next(iter(benchmark_methods))
        sorted_tables = sorted(
            [t for t in all_tables if t != "avg"],
            key=lambda t: dubug_logger.join_cost_distributions[benchmark][primary_method].get(t, 0),
            reverse=True
        )
        
        # 添加平均值行
        if "avg" in all_tables:
            sorted_tables.append("avg")
        
        # 为每个表添加行
        for i, table_name in enumerate(sorted_tables):
            row = []
            # 只在该基准测试的第一行显示基准测试名称
            if i == 0:
                row.append(benchmark)
            else:
                row.append("")
            
            # 添加表名
            row.append(table_name)
            
            # 添加每个方法的成本
            for method in method_list:
                if method in benchmark_methods:
                    cost = dubug_logger.join_cost_distributions[benchmark][method].get(table_name, "N/A")
                    row.append(f"{cost:,}" if isinstance(cost, (int, float)) else cost)
                else:
                    row.append("N/A")
            
            # 如果是平均值行，使用粗体样式
            if table_name == "avg":
                table.add_row(*row, style="bold")
            else:
                table.add_row(*row)
    
    # 打印表格
    console.print("\n[bold yellow]Join Cost Distributions By Method[/bold yellow]\n")
    console.print(table)

def get_method_style(method):
    """返回每个方法的颜色样式"""
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
    # 能不能改成多线程，这样可以同时运行多个实验

    # 配置日志
    # logging.basicConfig(filename=f'{base_dir}/../logs/column_selection_experiment.log',
    #                     level=logging.INFO,
    #                     format='%(asctime)s - %(levelname)s - %(message)s')

    using_join_order_strategy='traditional' if disabled_prim_reorder else 'prim'   # MTO的prim形式才能显示真实的计划
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
    # for item in [1000, 5000, 10000, 20000,50000]:
    for item in [1000, 5000, 20000, 50000]:
        settings.block_size=item
        run_evaluation(save_result=True, disabled_prim_reorder=True)


def scaling_replication():
    # for item in [1,2,3,4,5]:
    for item in [1,2,3,4]:
        settings.replication_factor=item
        run_evaluation(save_result=True, disabled_prim_reorder=False)
        

def test_model_overhead_by_scaling_scale():
    # 本质上，也是根据scale-factor来调整block-size的大小
    # 1G=>10000
    # 10G=>1000  因为我目前只有1G数据， 通过缩放来判断。 但是访问的每个块大小要乘以系数  scaling_ratio
    for sf in [5,10,20,50]:  #1,5,10,20,50
        settings.scale_factor=sf
        settings.block_size=int(10000/sf)
        run_evaluation(save_result=True, disabled_prim_reorder=False)
        # 将最终结果： 块字节数*scaling_ratio 作为真实的块字节数


if args.command == 0:  #基础实验
    settings.replication_factor=5
    run_evaluation(save_result=True, disabled_prim_reorder=False)
elif args.command == 1: # 扩展实验
    scaling_block_size()
elif args.command == 2:
    scaling_replication()
elif args.command == 3:
    test_model_overhead_by_scaling_scale()
else:
    print("Invalid command")


# 待做项：
# 1. 优化prim算法，主要是考虑加入更多的初始节点（定义更多的启发式规则）
# (已解决) 2.检查一下opt time的构成，目前看，为什么paw的时间成本竟然高于mto…（难道join树构建更快？） ，另外，group time是不是目前影响较小？？？
# (已解决) 3.opt time是不是要分成两部分：树构建时间和 查询时的search和group时间，才更加合理一些？


"""
问题1：部分query，MTO大于PAW
因为此时，MTO为fake join，且谓词不集中于主键，而是次要键。（如movie_company的主键是id，还有一个键是movie_id）

问题2：部分query,PAC-Tree大于MTO
因为prim算法无法保证全局最优，所以hyper join能带来最小的预估shuffle量减少，但没有考虑到
多表实际执行时的shuffle成本。所以存在多估的情况。



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
