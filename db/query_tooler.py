import re
from connector import Connector
import argparse
import os
import copy
import pickle
from collections import defaultdict
import random

parser = argparse.ArgumentParser()
parser.add_argument('--benchmark', type=str, default='tpch', help='The benchmark to use (tpch, imdb, or join-order)')
# format queries命令 action_store_true
parser.add_argument('--format', action='store_true', help='Whether to format queries')
parser.add_argument('--export_mto_paw_queries', action='store_true', help='Whether to export paw and mto queries')
parser.add_argument('--export_pac_queries', action='store_true', help='Whether to export mto queries')


args=parser.parse_args()

current_path=os.path.dirname(os.path.abspath(__file__))

benchmark = args.benchmark
query_base_dir=current_path+'/../queryset/'+benchmark
dataset_base_dir=current_path+'/../dataset/'+benchmark



def formatting_query_seed():
    # Part 2: For each query, get scanned rows and calculate accessed column ranges
    connector = Connector(benchmark)
    connector.connect()
    
    schema_details=pickle.load(open(f'{dataset_base_dir}/metadata.pkl', 'rb'))
    
    queries,query_column_ranges = [],{}

    if os.path.exists(f'{query_base_dir}/seed1.txt'):
        
        # Read the content of seed1.txt
        with open(f'{query_base_dir}/seed1.txt', 'r') as file:
            content = file.read()

        # Extract queries from the content
        pattern = r'(?:^|\n)(?:select|create view).*?;(?:\n|$)'
        queries = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
    elif os.path.exists(f'{query_base_dir}/seed1'):
        # 遍历该目录下的所有sql文件，存入到queries变量中
        for file in os.listdir(f'{query_base_dir}/seed1'):
            if file.endswith('.sql'):
                with open(f'{query_base_dir}/seed1/{file}', 'r') as f:
                    queries.append(f.read())

    for i, query in enumerate(queries, 1):
        # Remove comments
        query_clean = re.sub(r'--.*', '', query)
        query_clean = query_clean.strip()
        
        # Skip empty queries
        if not query_clean:
            continue

        # if i!=8:
        #     continue

        print(f"\nProcessing Query {i}:\n{query_clean}\n")
        
        # Use EXPLAIN (FORMAT JSON) to get the query plan
        explain_query = f"EXPLAIN (FORMAT JSON) {query_clean}"
        explain_result = connector.execute_query(explain_query)
        # The result is a list of tuples; get the first element
        plan_json = explain_result[0][0]
        # Parse the JSON plan
        plan = plan_json[0]['Plan']

        # Function to remove alias prefixes from predicates
        def remove_alias_prefix(pred):
            return re.sub(r'\b\w+\.', '', pred)

        # Function to filter out predicates with parameter placeholders
        def filter_predicates(preds):
            return [pred for pred in preds if not re.search(r'\$\d+', pred) and pred.strip()]

        # Function to replace aliases in predicates
        def replace_aliases(pred, aliases):
            for alias in aliases:
                pred = re.sub(r'\b{}\.'.format(re.escape(alias)), '', pred)
            return pred

        # Function to extract and process predicates
        def process_predicates(table, preds, aliases):
            filtered_preds = []
            for pred in preds:
                pred = replace_aliases(pred, aliases)
                if not re.search(r'\$\d+', pred) and pred.strip():
                    filtered_preds.append(pred)
            return filtered_preds

        def if_equal_pred(pred):
            pred=pred.strip().replace('(','').replace(')','').replace(' ','')
            if len(pred.split('='))!=2:
                return False
            col1,col2=pred.split('=')  # 判断col1和col2是否都是列名， 不带.
            # 依据schema_details 判断col1和col2是否都是列名
            if_col1,if_col2=False,False
            for table in schema_details:
                for col in schema_details[table]['numeric_columns']+schema_details[table]['text_columns']:
                    if col1==col:
                        if_col1=True
                    if col2==col:
                        if_col2=True
            if if_col1 and if_col2:
                return True
            return False

        # Function to extract tables, aliases, and predicates from the plan
        def extract_info(plan_node, tables_set, aliases_map, table_predicates):
            current_table = None
            current_alias = None
            # Get the relation name (table name) and alias if available
            if 'Relation Name' in plan_node:
                current_table = plan_node['Relation Name']
                tables_set.add(current_table)
                if 'Alias' in plan_node:
                    current_alias = plan_node['Alias']
                    aliases_map[current_alias.lower()] = current_table
            # Get the filter (predicate) if available
            if 'Filter' in plan_node and (current_table or current_alias):
                filter_text = plan_node['Filter']
                # Ignore internal predicates like '(NOT (hashed SubPlan 1))'
                if 'SubPlan' not in filter_text:
                    # Assign predicate to the current_table
                    if current_table:
                        # Remove alias prefix from predicates
                        pred = remove_alias_prefix(filter_text)
                        # Skip predicates containing parameter placeholders like $0  or two column condtions like a=b
                        if not re.search(r'\$\d+', pred) and not if_equal_pred(pred) and pred.strip():
                            table_predicates[current_table].append(pred)
            # Replace table aliases in predicates
            for table, preds in table_predicates.items():
                aliases = [alias for alias, tbl in aliases_map.items() if tbl == table]
                table_predicates[table] = process_predicates(table, preds, aliases)
            if 'Plans' in plan_node:
                for subplan in plan_node['Plans']:
                    extract_info(subplan, tables_set, aliases_map, table_predicates)

        def parse_join_condition(cond):
            """
            从 '(part.p_partkey = partsupp.ps_partkey)' 等形式中提取具体列，忽略 (SubPlan x)=xxx 的内容
            最终返回 ['p_partkey', 'ps_partkey'] 这样的列表
            """
            import re
            join_keys = []
            if not cond:
                return join_keys
            
            # 去除 '(SubPlan x) = ...' 这类部分
            cond = re.sub(r'\(SubPlan\s+\d+\)\s*=\s*\S+', '', cond)

            # 去掉外层括号、多余空白
            text = cond.strip('() ')

            # 查找形如 'table.col = alias.col' 或 'col = table.col' 的片段
            # pattern = r'(\w+(?:\.\w+)?)\s*=\s*(\w+(?:\.\w+)?)'
            # pairs = re.findall(pattern, text)
            
            # for left_col, right_col in pairs:
            #     # 去掉表或别名前缀，只保留最终列名
            pattern = r'((?:\w+\.)?\w+)\s*=\s*((?:\w+\.)?\w+)'
            matches = re.findall(pattern, text)
    
            for left, right in matches:
                left_parts = left.split('.') if '.' in left else [None, left]
                right_parts = right.split('.') if '.' in right else [None, right]
                
                left_table = left_parts[0]
                left_col = left_parts[-1]
                right_table = right_parts[0]
                right_col = right_parts[-1]
                
                # 验证列名是否有效
                left_valid = is_valid_column(left_col, left_table, schema_details)
                right_valid = is_valid_column(right_col, right_table, schema_details)
                
                if not (left_valid and right_valid):
                    continue  # 跳过无效的列
                
                # 处理两边都有表名的情况
                if left_table and right_table:
                    # 处理别名映射
                    if left_table in aliases_map:
                        left_table = aliases_map[left_table]
                    if right_table in aliases_map:
                        right_table = aliases_map[right_table]
                    join_keys.append(f"{left_table}.{left_col}={right_table}.{right_col}")
                
                # 处理左边没有表名的情况
                elif not left_table and right_table:
                    # 处理别名映射
                    if right_table in aliases_map:
                        right_table = aliases_map[right_table]
                    
                    # 查找左列所属的表
                    left_table = find_table_for_column(left_col, tables_set, schema_details)
                    if left_table:
                        join_keys.append(f"{left_table}.{left_col}={right_table}.{right_col}")
                
                # 处理右边没有表名的情况
                elif left_table and not right_table:
                    # 处理别名映射
                    if left_table in aliases_map:
                        left_table = aliases_map[left_table]
                    
                    # 查找右列所属的表
                    right_table = find_table_for_column(right_col, tables_set, schema_details)
                    if right_table:
                        join_keys.append(f"{left_table}.{left_col}={right_table}.{right_col}")
            
            return join_keys

        
        def is_valid_column(col_name, table_name=None, schema_details=None):
            """
            验证列名是否存在于模式中
            """
            if not schema_details:
                return True  # 如果没有模式信息，默认为有效
                
            # 如果提供了表名，只检查该表
            if table_name:
                if table_name in schema_details:
                    numeric_cols = schema_details[table_name].get('numeric_columns', [])
                    text_cols = schema_details[table_name].get('text_columns', [])
                    return col_name in numeric_cols or col_name in text_cols
                return False  # 表不存在于模式中
                
            # 如果没有提供表名，检查所有表
            for table in schema_details:
                numeric_cols = schema_details[table].get('numeric_columns', [])
                text_cols = schema_details[table].get('text_columns', [])
                if col_name in numeric_cols or col_name in text_cols:
                    return True
                    
            return False  # 列名不存在于任何表中
        

        def find_table_for_column(column, tables_set, schema_details):
            """
            根据列名查找所属的表
            """
            candidate_tables = []
            for table in tables_set:
                if table in schema_details:
                    numeric_cols = schema_details[table].get('numeric_columns', [])
                    text_cols = schema_details[table].get('text_columns', [])
                    
                    if column in numeric_cols or column in text_cols:
                        candidate_tables.append(table)
            
            # 如果只找到一个表，返回它
            if len(candidate_tables) == 1:
                return candidate_tables[0]
            
            # 如果找到多个表，可以基于上下文或查询执行计划选择最可能的一个
            # 这里简单地返回第一个，实际应用中可能需要更复杂的逻辑
            elif len(candidate_tables) > 1:
                return candidate_tables[0]
            
            return None

        def extract_joins_from_plan(plan_node, join_info, visited_tables,table_predicates):
            """
            递归提取Join相关信息，记录实际使用到的表、join_keys数组
            """
            node_type = plan_node.get('Node Type', '')
            if 'Join' in node_type:
                cond = plan_node.get('Hash Cond') or plan_node.get('Merge Cond') or plan_node.get('Join Filter')
                join_keys = parse_join_condition(cond)
                if join_keys:
                    for key in join_keys:
                        join_info.append({
                            'join_type': node_type,
                            'join_keys': [key]
                        })
            if 'Join Filter' in plan_node:
                join_keys = parse_join_condition(plan_node['Join Filter'])
                for key in join_keys:
                    join_info.append({
                        'join_type': node_type,
                        'join_keys': [key]
                    })
            if 'Index Cond' in plan_node:
                index_conds=plan_node['Index Cond'].split(' AND ')
                for index_cond in index_conds:
                    index_cond=index_cond.replace('((','(').replace('))',')')
                    if '.' not in index_cond:
                        table_predicates[plan_node['Relation Name']].append(index_cond)
                    else:
                        join_keys = parse_join_condition(index_cond)
                        # join_key_list=join_keys[0].split('.')
                        # join_key_list[0]=plan_node['Relation Name']
                        # new_join_keys=['.'.join(join_key_list)]
                        if join_keys:
                            join_info.append({
                                'join_type': 'Index Cond',
                                'join_keys': join_keys
                            })
            # 继续深入子节点
            if 'Plans' in plan_node:
                for child in plan_node['Plans']:
                    extract_joins_from_plan(child, join_info, visited_tables,table_predicates)
        
        
        # Sets to store tables, aliases, and a dictionary for table-specific predicates
        tables_set = set()
        aliases_map = {}
        table_predicates = defaultdict(list)
        # Extract information from the plan
        extract_info(plan, tables_set, aliases_map, table_predicates)
        table_list = list(tables_set)


        join_info = []
        visited = set()
        extract_joins_from_plan(plan, join_info, visited,table_predicates)
        
        # Build and execute SQL queries to get min and max for numeric columns
        ranges = {}
        for table in table_list:
            if table not in schema_details:
                continue
            numeric_cols = schema_details[table]['numeric_columns']
            if not numeric_cols:
                continue
            # Retrieve predicates specific to the current table
            if table not in table_predicates:
                continue
            predicates = table_predicates.get(table, [])
            where_clause = ' AND '.join(predicates)

            # Construct MIN and MAX queries for each numeric column
            min_max_selects = [f"MIN({col}) as min_{col}, MAX({col}) as max_{col}" for col in numeric_cols]
            
            min_max_query=f"SELECT {', '.join(min_max_selects)} FROM {table} WHERE {where_clause};"
            # min_max_query=f"SELECT {','.join(numeric_cols)} FROM {table} WHERE {where_clause} order by {numeric_cols[0]};"
            
            print(f"Executing min/max query on table '{table}':\n{min_max_query}\n")


            # Execute the query
            try:
                min_max_result = connector.execute_query(min_max_query)
                if min_max_result[0][0]:
                    col_ranges = {}
                    result_row = min_max_result[0]
                    num_cols = len(numeric_cols)
                    for idx, col in enumerate(numeric_cols):
                        min_val = result_row[idx * 2]
                        max_val = result_row[idx * 2 + 1]
                        if min_val is not None and max_val is not None:
                            col_ranges[col] = {'min': min_val, 'max': max_val}
                    ranges[table] = col_ranges
            except Exception as e:
                print(f"Error executing min/max query: {e} !!!!!!")

        
        
        if not ranges and not join_info:
            continue
        query_column_ranges[f'Query {i}'] = {
            'ranges': ranges,
            'join_info': join_info
        }
    # 保存query_column_ranges结果到.pkl文件
    with open(f'{query_base_dir}/formatted_queries.pkl', 'wb') as f:
        pickle.dump(query_column_ranges, f)
    # 同时，写一个副本，保存到txt文件中
    with open(f'{query_base_dir}/formatted_queries.txt', 'w') as f:
        for query_name, tables in query_column_ranges.items():
            f.write(f"{query_name}:\n")
            for table, cols in tables['ranges'].items():
                f.write(f"  Table: {table}\n")
                for col, rng in cols.items():
                    f.write(f"    {col}: min={rng['min']}, max={rng['max']}\n")
            for join in tables['join_info']:
                f.write(f"  Join Type: {join['join_type']}\n")
                f.write(f"  Join Keys: {join['join_keys']}\n")
            f.write("\n")

    connector.close_all_connection()


def validate_queries(method):
    saved_queries=pickle.load(open(f'{query_base_dir}/formatted_queries.pkl', 'rb'))
    schema_details=pickle.load(open(f'{dataset_base_dir}/metadata.pkl', 'rb'))
    
    # solved_queries={}
    for query_name, query_item in saved_queries.items():
        # if query_name!='Query 20':
        #     continue
        
        # 第一步，join-induced filters处理
        cnt=0
        while cnt<3: 
            for join_item in query_item['join_info']:
                # new_join_keys=[]
                # 需要循环2次，避免某些表未添加
                # for i in range(len(join_item['join_keys'])):
                left_table, left_col, right_table, right_col=join_item['join_keys'][0].replace('=','.').split('.')

                # check if left_table and right_table are in the same join
                if left_table in query_item['ranges'] and right_table in query_item['ranges']:
                    if left_col in query_item['ranges'][left_table] and right_col in query_item['ranges'][right_table]:
                        left_col_ranges=query_item['ranges'][left_table][left_col]
                        right_col_ranges=query_item['ranges'][right_table][right_col]
                        query_item['ranges'][left_table][left_col]['min']=max(left_col_ranges['min'], right_col_ranges['min'])
                        query_item['ranges'][right_table][right_col]['min']=max(left_col_ranges['min'], right_col_ranges['min'])
                        query_item['ranges'][left_table][left_col]['max']=min(left_col_ranges['max'], right_col_ranges['max'])
                        query_item['ranges'][right_table][right_col]['max']=min(left_col_ranges['max'], right_col_ranges['max'])
                    elif left_col in query_item['ranges'][left_table]:
                        left_col_ranges=query_item['ranges'][left_table][left_col]
                        query_item['ranges'][right_table][right_col]=left_col_ranges
                    elif right_col in query_item['ranges'][right_table]:
                        right_col_ranges=query_item['ranges'][right_table][right_col]
                        query_item['ranges'][left_table][right_col]=right_col_ranges
                
                elif left_table in query_item['ranges'] and right_table not in query_item['ranges']:
                    if left_col in query_item['ranges'][left_table]:
                        left_col_ranges=query_item['ranges'][left_table][left_col]
                        query_item['ranges'][right_table]={
                            right_col:left_col_ranges
                        }
                        for col in schema_details[right_table]['numeric_columns']:
                            if col!=right_col:
                                query_item['ranges'][right_table][col]=schema_details[right_table]['ranges'][col]
                        
                elif left_table not in query_item['ranges'] and right_table in query_item['ranges']:
                    if right_col in query_item['ranges'][right_table]:
                        right_col_ranges=query_item['ranges'][right_table][right_col]
                        query_item['ranges'][left_table]={
                            left_col:right_col_ranges
                        }
                        for col in schema_details[left_table]['numeric_columns']:
                            if col!=left_col:
                                query_item['ranges'][left_table][col]=schema_details[left_table]['ranges'][col]
                else:
                    continue    
            cnt+=1

        new_join_items=[]
        join_record=dict()
        for join_item in query_item['join_info']:
            
            left_table_col, right_table_col=join_item['join_keys'][0].split('=')
            left_table, left_col=left_table_col.split('.')
            right_table, right_col=right_table_col.split('.')
            if left_table_col==right_table_col:
                continue
            if left_table in join_record and join_record[left_table]==right_table:
                continue

            join_record[left_table]=right_table
            join_record[right_table]=left_table

            new_join_items.append(join_item)
        if method=="mto":
            random.shuffle(new_join_items)
        query_item['join_info']=new_join_items
        
        ranges=copy.deepcopy(query_item['ranges'])     
        for table, col_ranges in ranges.items():
            new_col_ranges={}
            for col, rng in col_ranges.items():
                if rng['min']!=schema_details[table]['ranges'][col]['min'] or rng['max']!=schema_details[table]['ranges'][col]['max']:
                    new_col_ranges[col]=rng
            query_item['ranges'][table]=new_col_ranges
            
            
    with open(f'{query_base_dir}/{method}_queries.pkl', 'wb') as f:
        pickle.dump(saved_queries, f)

    # with open(f'{query_base_dir}/{method}_queries.txt', 'w') as f:
    #     for query_name, query_item in saved_queries.items():
    #         f.write(f"{query_name}:\n")
    #         for table, cols in query_item['ranges'].items():
    #             f.write(f"  Table: {table}\n")
    #             for col, rng in cols.items():
    #                 f.write(f"    {col}: min={rng['min']}, max={rng['max']}\n")
    #         for join in query_item['join_info']:
    #             f.write(f"  Join Type: {join['join_type']}\n")
    #             f.write(f"  Join Keys: {join['join_keys']}\n")
    #         f.write("\n")
        
def paw_queries():
    formatted_queries=pickle.load(open(f'{query_base_dir}/formatted_queries.pkl', 'rb'))
    verified_queries=pickle.load(open(f'{query_base_dir}/mto_queries.pkl', 'rb'))
    paw_queries={}
    for query_name, query_item in formatted_queries.items():
        paw_queries[query_name]={
            'ranges':query_item['ranges'],
            'join_info':verified_queries[query_name]['join_info']
        }

    with open(f'{query_base_dir}/paw_queries.pkl', 'wb') as f:
        pickle.dump(paw_queries, f)

args.format=True
args.export_mto_paw_queries=True
args.export_pac_queries=True

if args.format:
    formatting_query_seed()

if args.export_mto_paw_queries:
    # # 该函数不仅仅是验证，而是对查询进行了一些处理，比如join-induced filters处理（这对后续的join order算法至关重要。传统join-indcued的谓词，才能准确地估计各表潜在的join成本，
    # # 并且构建更好的join tree。
    validate_queries(method="mto")
    paw_queries()

if args.export_pac_queries:
    validate_queries(method="pac")
