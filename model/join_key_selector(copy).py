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
import random
from join_eval import JoinEvaluator
import os
from join_order import JoinGraph,TraditionalJoinOrder
import pandas as pd
from db.conf import table_suffix

base_dir=os.path.dirname(os.path.abspath(__file__))
benchmark='imdb'  #'tpch' imdb
block_size=10000

class TwoLayerTree:
    def __init__(self,trees):
        self.candidate_trees=trees

class QdTree:
    def __init__(self,trees):
        self.candidate_trees=trees 
        
class TpchHypergraph:
    def __init__(self):
        """
        Each key in 'nodes' is a TPC-H table name (hypernode),
        and the value is a list of candidate columns (nodes within hypernode).
        """
        self.benchmark = 'tpch'
        self.candidate_nodes = {
            "nation": ["n_nationkey","n_regionkey"],
            "region": ["r_regionkey"],
            "customer": ["c_custkey", "c_nationkey"],
            "orders": ["o_orderkey", "o_custkey"],
            "lineitem": ["l_partkey","l_orderkey","l_suppkey"],
            "part": ["p_partkey"],
            "supplier": ["s_suppkey", "s_nationkey"],
            "partsupp": ["ps_suppkey","ps_partkey"]
        }
        
        self.hyper_nodes=  {table:[] for table in self.candidate_nodes.keys()}
        
    def clear_hyper_nodes(self):
        self.hyper_nodes=  {table:[] for table in self.candidate_nodes.keys()}

    def show_selections(self):
        for table, columns in self.hyper_nodes.items():
            print(f"Hypernode: {table}, Selected Columns: {columns}")

class JobHypergraph:
    def __init__(self):
        """
        Each key in 'nodes' is a JOB table name (hypernode),
        and the value is a list of candidate columns (nodes within hypernode).
        """
        self.benchmark = 'imdb'
        # 该候选节点，仅给出每个表潜在的join key列
        # 注意是job benchmark
        self.candidate_nodes = {
            "aka_name": ["id","person_id"],
            'aka_title': ["id","movie_id","kind_id"],
            "cast_info": ["id", "person_id", "movie_id", "person_role_id",  "role_id"],
            "char_name": ["id"],
            "comp_cast_type": ["id"],
            "company_name": ["id"],
            "company_type": ["id"],
            "complete_cast": ["id","movie_id","subject_id","status_id"],
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
        
        self.hyper_nodes=  {table:[] for table in self.candidate_nodes.keys()}
        
    def clear_hyper_nodes(self):
        self.hyper_nodes=  {table:[] for table in self.candidate_nodes.keys()}

    def show_selections(self):
        for table, columns in self.hyper_nodes.items():
            print(f"Hypernode: {table}, Selected Columns: {columns}")

# 为所有表创建候选join trees
def create_join_trees(hypergraph):
    tree_dict={}
    pa=PartitionAlgorithm(benchmark=hypergraph.benchmark)
    pa.load_join_query(using_example=False)
    for table, columns in hypergraph.candidate_nodes.items():
        tree_dict[table]={}
        # pa.load_join_query()  #目前，需要给一个比较合理的样本
        for column in columns:
            pa.table_name=table
            pa.load_data()
            pa.load_query(using_example=False)
            pa.LogicalJoinTree(join_col=column)
            tree_dict[table][column]=pa.partition_tree
    return TwoLayerTree(tree_dict),pa.join_queries


def create_qdtrees():
    tree_dict={}
    pa=PartitionAlgorithm(benchmark=hypergraph.benchmark)
    pa.load_join_query(using_example=False)
    for table, _ in hypergraph.candidate_nodes.items():
        # pa.load_join_query()  #目前，需要给一个比较合理的样本
        pa.table_name=table
        pa.load_data()
        pa.load_query(using_example=False)
        pa.InitializeWithQDT()
        tree_dict[table]=pa.partition_tree
    return QdTree(tree_dict),pa.join_queries

# Build hypergraph structure
hypergraph=None
if benchmark=='tpch':
    hypergraph = TpchHypergraph() 
elif benchmark=='imdb':
    hypergraph = JobHypergraph()

# 直接读取 joinTreer, join_queries等关键信息

# # Create join trees for all tables
# joinTreer = create_join_trees(hypergraph)
# with open(f'{base_dir}/../layouts/{hypergraph.benchmark}/join-trees.pkl','wb') as f:
#     pickle.dump(joinTreer,f)

# # # Create QD trees for all tables
# qdTreer,join_queries = create_qdtrees()
# with open(f'{base_dir}/../layouts/{hypergraph.benchmark}/qd-trees.pkl','wb') as f:
#     pickle.dump((qdTreer,join_queries),f)

# exit(-1)

joinTreer,_=pickle.load(open(f'{base_dir}/../layouts/{benchmark}/join-trees.pkl','rb'))
qdTreer,join_queries=pickle.load(open(f'{base_dir}/../layouts/{benchmark}/qd-trees.pkl','rb'))

# ~~~~~~~~读取表的部分元数据信息~~~~~~~~~
metadata={}
for table,tree in joinTreer.candidate_trees.items():
    sample_tree=list(tree.values())[0]
    metadata[table]={'row_count':sample_tree.pt_root.node_size,'used_cols':sample_tree.used_columns}    
table_metadata=pickle.load(open(f'{base_dir}/../dataset/{benchmark}/metadata.pkl','rb'))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def estimate_join_cost(hypergraph,current_table):
    shuffle_times,hyper_times={},{}
    join_cost=0
    for q_id, join_query in enumerate(join_queries):
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
            join_cost+=left_cost+right_cost if is_hyper else (left_cost+right_cost)*3
    
    return join_cost

# 现在，我已经给出了每个超节点初步的候选可以构建超图的边结构了。
# 定义函数：可以大致估算超图中，该表所在边的潜在join成本
def cpt_edge_join_information(hypergraph,group_type,join_strategy='traditional',current_table=None,single_layter_tree=False):
    # 计算当前超图中hyper边和shuffle边的位置，以及估算的总成本
    shuffle_times,hyper_times={},{}
    join_cost_dict={}
    cost_log=[]
    byte_log=[]
    solved_tables={}
    for q_id,join_query in enumerate(join_queries):
        print('processing query:',q_id)
        # 计算hyper cost，便于确定join order
        if q_id>=3:
            continue
        query_cost_log,query_bytes_log=list(),list()
        solved_tables[q_id]=set()
        if join_query['join_relations']:
            if join_strategy=='traditional':
                join_path=TraditionalJoinOrder(join_query['join_relations'],metadata).paths
            else:
                scan_block_dict={'card':{},'relation':[]}
                for join_op in join_query['join_relations']:
                    left_table,left_col_idx=list(join_op.items())[0]
                    right_table,right_col_idx=list(join_op.items())[1]
                    left_col,right_col=metadata[left_table]["used_cols"][left_col_idx],metadata[right_table]["used_cols"][right_col_idx]
                    left_query=join_query['vectors'][left_table]
                    right_query=join_query['vectors'][right_table]
                    is_hyper=True
                    if single_layter_tree:
                        if f'{left_table}.{left_col}' not in scan_block_dict["card"]:
                            scan_block_dict["card"][f'{left_table}.{left_col}']=len(qdTreer.candidate_trees[left_table].query_single(left_query))
                        
                        if f'{right_table}.{right_col}' not in scan_block_dict["card"]:
                            scan_block_dict["card"][f'{right_table}.{right_col}']=len(qdTreer.candidate_trees[right_table].query_single(right_query))
                        is_hyper=False
                    else:
                        if left_col in hypergraph.hyper_nodes[left_table]:
                            tree=joinTreer.candidate_trees[left_table][left_col]
                        else:
                            # 此时，无法在被选择树中无法直接找到对应的join_tree，
                            tree=list(joinTreer.candidate_trees[left_table].values())[0]
                            is_hyper=False
                        if f'{left_table}.{left_col}' not in scan_block_dict["card"]:
                            scan_block_dict["card"][f'{left_table}.{left_col}']=len(tree.query_single(left_query))
                        
                        if right_col in hypergraph.hyper_nodes[right_table]:
                            tree=joinTreer.candidate_trees[right_table][right_col]
                        else:
                            tree=list(joinTreer.candidate_trees[right_table].values())[0]
                            is_hyper=False
                        if f'{right_table}.{right_col}' not in scan_block_dict["card"]:
                            scan_block_dict["card"][f'{right_table}.{right_col}']=len(tree.query_single(right_query))
                    
                    scan_block_dict["relation"].append([f'{left_table}.{left_col}',f'{right_table}.{right_col}','Hyper' if is_hyper else 'Shuffle'])
                
                jg=JoinGraph(scan_block_dict)
                join_path=jg.generate_MST()
            temp_table_dfs=[]
            temp_joined_bytes=[]
            temp_shuffle_ops=[]
            for order_id,item in enumerate(join_path):
                join_queryset,joined_cols,joined_trees,joined_tables=[],[],[],[]
                # temp_target_trees=[]
                is_hyper=True if item[2]==1 else False
                
                is_real_hyper=True
                join_ops=[(item[0].table,item[0].adj_col[item[1]]),(item[1].table,item[1].adj_col[item[0]])]
                
                for join_table,join_col in join_ops:
                    join_queryset.append(join_query['vectors'][join_table])
                    join_col_idx=metadata[join_table]["used_cols"].index(join_col)
                    joined_cols.append(join_col_idx)
                    if single_layter_tree:
                        tree=qdTreer.candidate_trees[join_table]
                        is_real_hyper=False
                    else:
                        if join_col in hypergraph.hyper_nodes[join_table]:
                            tree=joinTreer.candidate_trees[join_table][join_col]
                        else:
                            # 此时，无法在被选择树中无法直接找到对应的join_tree，
                            tree=list(joinTreer.candidate_trees[join_table].values())[0]
                            is_real_hyper=False
                    joined_trees.append(tree)
                    joined_tables.append(join_table)
                    # temp_target_trees.append(joinTreer.candidate_trees[join_table][join_col])
                
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
                    temp_joined_bytes.append(table_metadata[item[1].table]['read_line'])
                    continue
                
                
                # if current_table and current_table not in joined_tables:  #只获取当前表的代价信息
                #     continue
                # 确定主次表 （大表在左，小表在右）
                if len(joined_trees[0].query_single(join_queryset[0]))>len(joined_trees[1].query_single(join_queryset[1])):
                    join_queryset[0],join_queryset[1]=join_queryset[1],join_queryset[0]
                    joined_cols[0],joined_cols[1]=joined_cols[1],joined_cols[0]
                    joined_trees[0],joined_trees[1]=joined_trees[1],joined_trees[0]
                    joined_tables[0],joined_tables[1]=joined_tables[1],joined_tables[0]
                    # temp_target_trees[0],temp_target_trees[1]=temp_target_trees[1],temp_target_trees[0]
                
                left_table,right_table=joined_tables[0],joined_tables[1]
                if is_real_hyper:
                    hyper_times[left_table]=hyper_times.get(left_table,0)+1
                    hyper_times[right_table]=hyper_times.get(right_table,0)+1
                else:
                    shuffle_times[left_table]=shuffle_times.get(left_table,0)+1
                    shuffle_times[right_table]=shuffle_times.get(right_table,0)+1
                
                solved_tables[q_id].add(left_table)
                solved_tables[q_id].add(right_table)
                
                join_eval = JoinEvaluator(join_queryset,joined_cols,joined_trees,joined_tables,block_size,table_metadata,benchmark=hypergraph.benchmark)
                hyper_shuffle_cost,hyper_read_bytes, temp_joined_df=join_eval.rough_join_cost(group_type)
                join_cost_dict[left_table]=join_cost_dict.get(left_table,0)+hyper_shuffle_cost//2
                join_cost_dict[right_table]=join_cost_dict.get(right_table,0)+hyper_shuffle_cost//2
                query_cost_log.append(hyper_shuffle_cost)
                query_bytes_log.append(hyper_read_bytes)
                temp_table_dfs.append(temp_joined_df)
                temp_joined_bytes.append(table_metadata[left_table]['read_line']+table_metadata[right_table]['read_line'])
            final_joined_sizes=[df.shape[0] for df in temp_table_dfs]
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
                    shuffle_cost+=(temp_table_dfs[i].shape[0]+last_temp_dfs.shape[0])*3
                    shuffle_bytes+=(temp_table_dfs[i].shape[0]*last_temp_line+last_temp_dfs.shape[0]*temp_joined_bytes[i])*3
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
            # 无join操作，直接扫描表
            print('no join operation')
            final_joined_sizes=[]
            for table, query_vector in join_query['vectors'].items():
                solved_tables[q_id].add(table)
                if single_layter_tree:
                    tree=qdTreer.candidate_trees[table]
                else:
                    tree=list(joinTreer.candidate_trees[table].values())[0]
                b_ids=tree.query_single(query_vector)
                scan_cost=sum([tree.nid_node_dict[b_id].node_size for b_id in b_ids])
                
                join_cost_dict[table]=join_cost_dict.get(table,0)+scan_cost
                query_cost_log.append(scan_cost)
                query_bytes_log.append(table_metadata[table]['read_line']*scan_cost)
                final_joined_sizes.append(scan_cost)
        cost_log.append(query_cost_log)
        byte_log.append(query_bytes_log)
    
    query_cost_dict={}
    query_ratio_dict={}
    for qid in range(len(cost_log)):
        query_cost_dict[qid]=sum(list(cost_log[qid]))
        query_ratio_dict[qid]=sum(list(byte_log[qid]))/sum([sum(table_metadata[table]['width'].values())*table_metadata[table]['rows'] for table in solved_tables[qid]])
        print(f"Query {qid} cost log: {cost_log[qid]}")    
    print(f"Query ratio log: {query_ratio_dict}")    
    print('final joined sizes: ',final_joined_sizes)
    final_join_result={}
    for table in join_cost_dict.keys():
        final_join_result[table]={'shuffle times':shuffle_times.get(table,0),'hyper times':hyper_times.get(table,0),'join cost':join_cost_dict[table]}
    return final_join_result,query_cost_dict,query_ratio_dict


def select_join_columns_by_MTO(hypergraph,group_type=0,strategy='traditional'):

    final_join_result,query_cost_dict,query_ratio_dict=cpt_edge_join_information(hypergraph,group_type,join_strategy=strategy,single_layter_tree=True)
    print("Final Column Selection Cost of MTO is: ",sum(list(query_cost_dict.values()))/len(query_cost_dict.keys()))
    print("Average Data Cost Ratio of MTO is: ",sum(list(query_ratio_dict.values()))/len(query_ratio_dict.keys()))
    hypergraph.show_selections()
    hypergraph.clear_hyper_nodes()
    return final_join_result,query_cost_dict
    

def select_join_columns_by_ADP(hypergraph,group_type=0,strategy='traditional'):
    
    for table in hypergraph.hyper_nodes:
        if metadata[table]['used_cols'][0] in hypergraph.candidate_nodes[table]:
            hypergraph.hyper_nodes[table].append(metadata[table]['used_cols'][0])
    
    final_join_result,query_cost_dict,query_ratio_dict=cpt_edge_join_information(hypergraph,group_type,join_strategy=strategy)
    print("Final Column Selection Cost of ADP is: ",sum(list(query_cost_dict.values()))/len(query_cost_dict.keys()))
    print("Average Data Cost Ratio of ADP is: ",sum(list(query_ratio_dict.values()))/len(query_ratio_dict.keys()))
    # 返回最终的列选择结果
    hypergraph.show_selections()
    hypergraph.clear_hyper_nodes()
    return final_join_result,query_cost_dict

# 构建邻接矩阵，节点是列，边是两个列之间的join关系，权重是查询中两个列join的次数
# 选择列的时候，选择度数最大的列，然后选择与它相邻的列，直到所有列都被选择 
def select_join_columns_by_JT(hypergraph,group_type,replication_factor = 2,strategy='traditional'):

    # Step 1：Build adjacency matrix
    adj_matrix = {}
    # node_weights = {}
    for query in join_queries:
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
    
    # Step 2：根据邻接矩阵的权重，来分配初始列。
    
    # 为每个表分配初始列，选择连接度数最大的列，同时考虑每个度对应的表规模
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
    
        # 按照值重新排序 col_freq_dict，排序后的keys按降序顺序存放到list返回
        col_list=sorted(col_freq_dict.items(), key=lambda x: x[1], reverse=True)
    
        for i in range(1,replication_factor+1):
            if i > len(col_list):
                break
            hypergraph.hyper_nodes[table].append(col_list[i-1][0])

    
    
    # # 按照成本排序
    # 依此选择成本最高的表，通过调整分区键，来减少成本
    # 在当前选择结果基础上，可以尝试用其他候选列替换某个已有列，若能降低成本则采纳

    # def list_equal_list(list1,list2):
    #     if len(list1)!=len(list2):
    #         return False
    #     for i in range(len(list1)):
    #         if list1[i]!=list2[i]:
    #             return False
    #     return True
    
    # # for table_name, _cost_val in join_cost_dict:
    # for table_name in hypergraph.candidate_nodes:
    #     # if shuffle_times.get(table_name, 0) == 0:
    #     #     continue
    #     if list_equal_list(hypergraph.hyper_nodes[table_name],hypergraph.candidate_nodes[table_name]):
    #         continue
    #     current_cols = hypergraph.hyper_nodes[table_name]
    #     best_cols = current_cols[:]
    #     best_cost = estimate_join_cost(hypergraph,table_name)  # 返回table_name的当前cost
    #     # 对每个可用候选列，若未在current_cols中则尝试替换
    #     for candidate_col in hypergraph.candidate_nodes[table_name]:
    #         if candidate_col in current_cols:
    #             continue
    #         for replaced_col in current_cols:
    #             trial_cols = current_cols[:]
    #             idx = trial_cols.index(replaced_col)
    #             trial_cols[idx] = candidate_col
    #             # 临时修改为trial_cols，计算cost
    #             original_cols = hypergraph.hyper_nodes[table_name]
    #             hypergraph.hyper_nodes[table_name] = trial_cols
    #             new_cost = estimate_join_cost(hypergraph,table_name)
    #             # 如果新成本更低，更新best_cols
    #             if new_cost < best_cost:
    #                 best_cost = new_cost
    #                 best_cols = trial_cols[:]
    #             # 还原
    #             hypergraph.hyper_nodes[table_name] = original_cols
    #     # 如果找到更优方案，则更新
    #     hypergraph.hyper_nodes[table_name] = best_cols


    final_join_result,query_cost_dict,query_ratio_dict=cpt_edge_join_information(hypergraph,group_type,join_strategy=strategy)
    print("Final Column Selection Cost of JT is: ",sum(list(query_cost_dict.values()))/len(query_cost_dict.keys()))
    print("Average Data Cost Ratio of JT is: ",sum(list(query_ratio_dict.values()))/len(query_ratio_dict.keys()))
    # 返回最终的列选择结果
    hypergraph.show_selections()
    hypergraph.clear_hyper_nodes()
    return final_join_result,query_cost_dict

def save_experiment_result(experiment_result,replication_factor):
    # query_columns=['Model','Query ID','Join Cost']
    # query_data=[]
    # table_columns=['Model','Table','Shuffle Times','Hyper Times','Join Cost']
    # table_data=[]
    # for model, item in experiment_result.items():
    #     for q_id, cost in item[1].items():
    #         query_data.append([model,q_id,cost])
    #     for table,join_info in item[0].items():
    #         table_data.append([model,table,join_info['shuffle times'],join_info['hyper times'],join_info['join cost']])
        
    # query_df=pd.DataFrame(query_data,columns=query_columns)
    # table_df=pd.DataFrame(table_data,columns=table_columns)
    # query_df.to_csv(f'{base_dir}/../experiment/result/query_cost.csv',index=False)
    # table_df.to_csv(f'{base_dir}/../experiment/result/table_cost.csv',index=False)

    import pandas as pd
    import os

    base_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = f"{base_dir}/../experiment/result/{replication_factor}/"

    # 1) 从 experiment_result 中整理出 query_cost 和 table_cost 的 DataFrame
    # query_cost: Model, Query ID, Join Cost
    query_list = []
    table_list = []
    for model_name, (table_info, query_info) in experiment_result.items():
        # query_info: {q_id: cost}
        for q_id, cost_val in query_info.items():
            query_list.append([model_name, q_id, cost_val])
        # table_info: {table: {'shuffle times': x, 'hyper times': y, 'join cost': z}}
        for table, cost_dict in table_info.items():
            shuffle_times = cost_dict.get('shuffle times', 0)
            hyper_times = cost_dict.get('hyper times', 0)
            join_cost = cost_dict.get('join cost', 0)
            table_list.append([model_name, table, shuffle_times, hyper_times, join_cost])

    df_query = pd.DataFrame(query_list, columns=["Model", "Query ID", "Join Cost"])
    df_table = pd.DataFrame(table_list, columns=["Model", "Table", "Shuffle Times", "Hyper Times", "Join Cost"])

    # 2) 透视 df_query: 行是 Query ID，列是 Model，值是 Join Cost
    pivot_query = df_query.pivot(index="Query ID", columns="Model", values="Join Cost").reset_index()

    # 3) 对 df_table 做多次 pivot，再合并
    pivot_shuffle = df_table.pivot_table(index="Table", columns="Model", values="Shuffle Times", aggfunc="first").reset_index()
    pivot_hyper = df_table.pivot_table(index="Table", columns="Model", values="Hyper Times", aggfunc="first").reset_index()
    pivot_cost = df_table.pivot_table(index="Table", columns="Model", values="Join Cost", aggfunc="first").reset_index()

    pivot_shuffle.rename(
        columns={
            "ADP": "ADP_shuffle",
            "JT": "JT_shuffle",
            "MTO": "MTO_shuffle"
        },
        inplace=True
    )
    merged_df = pivot_shuffle.merge(pivot_hyper, on="Table", suffixes=(None, "_hyper"))
    merged_df = merged_df.merge(pivot_cost, on="Table", suffixes=(None, "_cost"))

    # 4) 保存结果到Excel
    os.makedirs(result_dir, exist_ok=True)
    with pd.ExcelWriter(f"{result_dir}/formatted_results.xlsx") as writer:
        pivot_query.to_excel(writer, sheet_name="Query_Cost", index=False)
        merged_df.to_excel(writer, sheet_name="Table_Cost", index=False)

def test_models(save_result=False,disabled_join=False):
    replication_factor = 2
    experiment_result={'MTO':[],'ADP':[],'JT':[]}
    if disabled_join:
        final_join_result,query_cost_dict=select_join_columns_by_MTO(hypergraph,strategy='traditional',group_type=0)
    else:
        final_join_result,query_cost_dict=select_join_columns_by_MTO(hypergraph,strategy='prim',group_type=0)
    experiment_result['MTO']=[final_join_result,query_cost_dict]
    
    if disabled_join:
        final_join_result,query_cost_dict=select_join_columns_by_ADP(hypergraph,strategy='traditional',group_type=0)
    else:
        final_join_result,query_cost_dict=select_join_columns_by_ADP(hypergraph,strategy='prim',group_type=0)
    experiment_result['ADP']=[final_join_result,query_cost_dict]
    
    final_join_result,query_cost_dict=select_join_columns_by_JT(hypergraph,group_type=1,strategy='prim',replication_factor=replication_factor)
    experiment_result['JT']=[final_join_result,query_cost_dict]
    if save_result:
        save_experiment_result(experiment_result,replication_factor) 

 
if __name__ == "__main__":
    test_models(disabled_join=True)
    