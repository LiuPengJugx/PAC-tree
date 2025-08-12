import numpy as np
import time
import pickle
import os
import datetime
import pandas as pd
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from model.partition_tree import PartitionTree
from model.join_eval import JoinEvaluator
from model.query_example import synetic_queries

base_dir = os.path.dirname(os.path.abspath(__file__))
"""
The core partitioning algorithm class 
Attribute: partition_tree is constructed partition tree.
"""


class PartitionAlgorithm:
    """
    The partition algorithms, inlcuding NORA, QdTree and kd-tree.
    """

    def __init__(self, block_size=10000, benchmark="tpch", table_name="lineitem"):
        self.block_size = block_size
        self.partition_tree = None
        self.benchmark = benchmark
        self.table_name = table_name
        self.machine_num = 10
        self.sample_rate = self.calculate_global_sample_rate()  # default 1

    def calculate_global_sample_rate(
        self, max_sample_size=1000000, dim_table_threshold=10000
    ):
        """
        计算整个数据库的统一抽样率，基于最大表的大小
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        metadata_path = f"{base_dir}/../dataset/{self.benchmark}/metadata.pkl"

        if not os.path.exists(metadata_path):
            return 1.0  # 如果没有元数据，返回1.0表示不抽样

        metadata = pickle.load(open(metadata_path, "rb"))
        max_rows = 0

        # 找出非维度表中最大的表
        for table, table_meta in metadata.items():
            rows = table_meta["rows"]
            # 跳过维度表（行数小于阈值的表）
            if rows > dim_table_threshold:
                max_rows = max(max_rows, rows)

        # 如果最大行数超过限制，计算抽样率；否则不抽样
        if max_rows > max_sample_size:
            return max_sample_size / max_rows
        return 1.0

    def load_data(self, dim_table_threshold=10000):
        """
        Load data from the data source.
        """
        # 读取表的阈值
        metadata = pickle.load(
            open(f"{base_dir}/../dataset/{self.benchmark}/metadata.pkl", "rb")
        )
        self.used_columns = metadata[self.table_name]["numeric_columns"]

        # 获取表域
        table_min_domains, table_max_domains = [], []
        for _, col_range in metadata[self.table_name]["ranges"].items():
            # datetime类型转化为'yyyy-mm-dd'的形式，其他类别不变
            table_min_domains.append(
                col_range["min"].strftime("%Y-%m-%d")
                if isinstance(col_range["min"], datetime.date)
                else float(col_range["min"])
            )
            table_max_domains.append(
                col_range["max"].strftime("%Y-%m-%d")
                if isinstance(col_range["max"], datetime.date)
                else float(col_range["max"])
            )
        self.table_domains = table_min_domains + table_max_domains
        self.column_width = metadata[self.table_name]["width"]
        total_rows = metadata[self.table_name]["rows"]

        # 读取csv数据
        file_path = f"{base_dir}/../dataset/{self.benchmark}/{self.table_name}.csv"
        if self.sample_rate < 1.0 and total_rows > dim_table_threshold:
            data = pd.read_csv(
                file_path,
                usecols=self.used_columns,
                skiprows=lambda i: i > 0 and np.random.random() > self.sample_rate,
            )
            print(
                f"对表 {self.table_name} 使用抽样: 总行数 {total_rows}，抽样 {len(data)} 行，抽样率 {self.sample_rate:.4f}"
            )
        else:
            data = pd.read_csv(file_path, usecols=self.used_columns)
        self.tabledata = data.values

    def load_join_query(self, using_example=False, join_indeuced="MTO"):
        """
        Load join query from the data source.
        """
        metadata = pickle.load(
            open(f"{base_dir}/../dataset/{self.benchmark}/metadata.pkl", "rb")
        )

        if using_example:
            query_dict = synetic_queries
        else:
            if join_indeuced == "MTO":
                query_dict = pickle.load(
                    open(
                        f"{base_dir}/../queryset/{self.benchmark}/mto_queries.pkl", "rb"
                    )
                )
            elif join_indeuced == "PAC":
                query_dict = pickle.load(
                    open(
                        f"{base_dir}/../queryset/{self.benchmark}/pac_queries.pkl", "rb"
                    )
                )
            else:
                query_dict = pickle.load(
                    open(
                        f"{base_dir}/../queryset/{self.benchmark}/paw_queries.pkl", "rb"
                    )
                )

        join_freqs = {table: {} for table in metadata.keys()}
        join_query_vectors = []
        # for queryid, item in query_dict.items():
        #     query_vectors={}
        #     for table, col_range in item['ranges'].items():
        #         min_vector,max_vector=[],[]
        #         for col in col_range:
        #             min_vector.append(
        #                 col_range[col]['min'].strftime('%Y-%m-%d') if isinstance(col_range[col]['min'], datetime.date) else col_range[col]['min'] if isinstance(col_range[col]['min'], str) else float(col_range[col]['min'])
        #             )
        #             max_vector.append(
        #                 col_range[col]['max'].strftime('%Y-%m-%d') if isinstance(col_range[col]['max'], datetime.date) else col_range[col]['max'] if isinstance(col_range[col]['max'], str) else float(col_range[col]['max'])
        #             )
        #         if min_vector:
        #             query_vectors[table]=min_vector+max_vector

        for queryid, item in query_dict.items():
            join_vector = {"vectors": [], "join_relations": []}
            for join_info in item["join_info"]:
                for join_preds in join_info["join_keys"]:
                    left_cond, right_cond = join_preds.split("=")
                    left_table, left_col = left_cond.split(".")
                    left_col = metadata[left_table]["numeric_columns"].index(left_col)
                    right_table, right_col = right_cond.split(".")
                    right_col = metadata[right_table]["numeric_columns"].index(
                        right_col
                    )
                    join_dict = {left_table: left_col, right_table: right_col}
                    if left_table not in item["ranges"]:
                        item["ranges"][left_table] = {}
                    if right_table not in item["ranges"]:
                        item["ranges"][right_table] = {}
                    join_freqs[left_table][left_col] = (
                        join_freqs[left_table].setdefault(left_col, 0) + 1
                    )
                    join_freqs[right_table][right_col] = (
                        join_freqs[right_table].setdefault(right_col, 0) + 1
                    )
                    if join_dict not in join_vector["join_relations"]:
                        join_vector["join_relations"].append(join_dict)

            query_vectors = {}
            for table, col_ranges in item["ranges"].items():
                min_vector, max_vector = [], []
                for col_name, col_domain in metadata[table]["ranges"].items():
                    if col_name not in col_ranges:
                        min_vector.append(
                            col_domain["min"].strftime("%Y-%m-%d")
                            if isinstance(col_domain["min"], datetime.date)
                            else (
                                col_domain["min"]
                                if isinstance(col_domain["min"], str)
                                else float(col_domain["min"])
                            )
                        )
                        max_vector.append(
                            col_domain["max"].strftime("%Y-%m-%d")
                            if isinstance(col_domain["max"], datetime.date)
                            else (
                                col_domain["max"]
                                if isinstance(col_domain["max"], str)
                                else float(col_domain["max"])
                            )
                        )
                    else:
                        min_vector.append(
                            col_ranges[col_name]["min"].strftime("%Y-%m-%d")
                            if isinstance(col_ranges[col_name]["min"], datetime.date)
                            else (
                                col_ranges[col_name]["min"]
                                if isinstance(col_ranges[col_name]["min"], str)
                                else float(col_ranges[col_name]["min"])
                            )
                        )
                        max_vector.append(
                            col_ranges[col_name]["max"].strftime("%Y-%m-%d")
                            if isinstance(col_ranges[col_name]["max"], datetime.date)
                            else (
                                col_ranges[col_name]["max"]
                                if isinstance(col_ranges[col_name]["max"], str)
                                else float(col_ranges[col_name]["max"])
                            )
                        )
                if min_vector:
                    query_vectors[table] = min_vector + max_vector

            join_vector["vectors"] = query_vectors

            join_query_vectors.append(join_vector)
        self.join_queries = join_query_vectors

        # join_freqs按照 value的值对key进行降序排列

        self.join_freqs = {
            table: dict(
                sorted(
                    join_freqs[table].items(), key=lambda item: item[1], reverse=True
                )
            )
            for table in join_freqs.keys()
        }

    def load_query(self, using_example=False, join_indeuced="MTO"):
        """
        Load query from the data source.
        """
        if using_example:
            query_dict = synetic_queries
        else:
            if join_indeuced == "MTO":
                query_dict = pickle.load(
                    open(
                        f"{base_dir}/../queryset/{self.benchmark}/mto_queries.pkl", "rb"
                    )
                )
            elif join_indeuced == "PAC":
                query_dict = pickle.load(
                    open(
                        f"{base_dir}/../queryset/{self.benchmark}/pac_queries.pkl", "rb"
                    )
                )
            else:
                query_dict = pickle.load(
                    open(
                        f"{base_dir}/../queryset/{self.benchmark}/paw_queries.pkl", "rb"
                    )
                )

        metadata = pickle.load(
            open(f"{base_dir}/../dataset/{self.benchmark}/metadata.pkl", "rb")
        )

        query_vectors = []

        # for queryid, item in query_dict.items():
        #     min_vector,max_vector=[],[]
        #     for table, col_range in item['ranges'].items():
        #         if table == self.table_name:
        #             for col in col_range:
        #                 min_vector.append(
        #                     col_range[col]['min'].strftime('%Y-%m-%d') if isinstance(col_range[col]['min'], datetime.date) else col_range[col]['min'] if isinstance(col_range[col]['min'], str) else float(col_range[col]['min'])
        #                 )
        #                 max_vector.append(
        #                     col_range[col]['max'].strftime('%Y-%m-%d') if isinstance(col_range[col]['max'], datetime.date) else col_range[col]['max'] if isinstance(col_range[col]['max'], str) else float(col_range[col]['max'])
        #                 )
        #         if min_vector: query_vectors.append(min_vector+max_vector)

        for queryid, item in query_dict.items():
            min_vector, max_vector = [], []
            for table, col_ranges in item["ranges"].items():
                if table != self.table_name:
                    continue
                for col_name, col_domain in metadata[table]["ranges"].items():
                    if col_name not in col_ranges:
                        min_vector.append(
                            col_domain["min"].strftime("%Y-%m-%d")
                            if isinstance(col_domain["min"], datetime.date)
                            else (
                                col_domain["min"]
                                if isinstance(col_domain["min"], str)
                                else float(col_domain["min"])
                            )
                        )
                        max_vector.append(
                            col_domain["max"].strftime("%Y-%m-%d")
                            if isinstance(col_domain["max"], datetime.date)
                            else (
                                col_domain["max"]
                                if isinstance(col_domain["max"], str)
                                else float(col_domain["max"])
                            )
                        )
                    else:
                        min_vector.append(
                            col_ranges[col_name]["min"].strftime("%Y-%m-%d")
                            if isinstance(col_ranges[col_name]["min"], datetime.date)
                            else (
                                col_ranges[col_name]["min"]
                                if isinstance(col_ranges[col_name]["min"], str)
                                else float(col_ranges[col_name]["min"])
                            )
                        )
                        max_vector.append(
                            col_ranges[col_name]["max"].strftime("%Y-%m-%d")
                            if isinstance(col_ranges[col_name]["max"], datetime.date)
                            else (
                                col_ranges[col_name]["max"]
                                if isinstance(col_ranges[col_name]["max"], str)
                                else float(col_ranges[col_name]["max"])
                            )
                        )
                if min_vector:
                    query_vectors.append(min_vector + max_vector)
            # join_vector=[]
            # for join_info in item['join_info']:
            #     for join_preds in join_info['join_keys']:
            #         left_cond,right_cond=join_preds.split('=')
            #         left_table,left_col=left_cond.split('.')
            #         right_table,right_col=right_cond.split('.')
            #         if left_table==table_name:
            #             join_vector.append(self.used_columns.index(left_col))
            #         if right_table==table_name:
            #             join_vector.append(self.used_columns.index(right_col))
            # if min_vector:
            #     join_query_vectors.append(list(set(join_vector)))
        self.queries = query_vectors
        # self.join_queries = join_query_vectors

    def eval_len_node(self, dataset):
        return int(len(dataset) // self.sample_rate)

    # MTO layout
    def InitializeWithQDT(self):
        """
        # should I also store the candidate cut positions in Partition Node ?
        The dimension of queries should match the dimension of boundary and dataset!
        """
        start_time = time.time()
        num_dims = len(self.used_columns)
        boundary = self.table_domains
        self.partition_tree = PartitionTree(num_dims, boundary, self.sample_rate)
        self.partition_tree.name = "QDT"
        self.partition_tree.used_columns = self.used_columns
        self.partition_tree.column_width = self.column_width
        self.partition_tree.pt_root.node_size = self.eval_len_node(self.tabledata)
        self.partition_tree.pt_root.dataset = self.tabledata
        self.partition_tree.pt_root.queryset = (
            self.queries
        )  # assume all queries overlap with the boundary
        self.__QDT(self.block_size)
        end_time = time.time()
        print(f"{self.table_name} Build Time (s):", end_time - start_time)
        self.partition_tree.build_time = end_time - start_time
        self.partition_tree.save_tree(
            f"{base_dir}/../layouts/bs={self.block_size}/{self.benchmark}/{self.table_name}-QDT.pkl"
        )

    def __QDT(self, block_size):
        """
        the QdTree partition algorithm
        """
        CanSplit = True
        print_s = True
        while CanSplit:
            CanSplit = False

            # for leaf in self.partition_tree.get_leaves():
            leaves = self.partition_tree.get_leaves()
            # print("# number of leaf nodes:",len(leaves))
            for leaf in leaves:

                # print("current leaf node id:",leaf.nid, "leaf node dataset size:",len(leaf.dataset))
                if leaf.node_size < 2 * block_size:
                    continue

                candidate_cuts = leaf.get_candidate_cuts()

                # get best candidate cut position
                skip, max_skip, max_skip_split_dim, max_skip_split_value = 0, -1, 0, 0
                for split_dim, split_value in candidate_cuts:

                    valid, skip, _, _ = leaf.if_split(
                        split_dim, split_value, block_size, self.sample_rate
                    )
                    if valid and skip > max_skip:
                        max_skip = skip
                        max_skip_split_dim = split_dim
                        max_skip_split_value = split_value

                if max_skip >= 0:
                    # if the cost become smaller, apply the cut
                    if print_s:
                        print("QDTREE CUT!")
                        print_s = False
                    child_node1, child_node2 = self.partition_tree.apply_split(
                        leaf.nid, max_skip_split_dim, max_skip_split_value
                    )
                    # print(" Split on node id:", leaf.nid)
                    CanSplit = True

    def LogicalJoinTree(self, join_col, join_max_depth=3):
        """
        # should I also store the candidate cut positions in Partition Node ?
        The dimension of queries should match the dimension of boundary and dataset!
        """
        start_time = time.time()
        logical_tree_path = (
            f"{base_dir}/../layouts/bs={self.block_size}/{self.benchmark}/logical"
        )
        os.makedirs(logical_tree_path, exist_ok=True)
        tree_path = f"{logical_tree_path}/{self.table_name}-{join_col}-tree.pkl"
        # if not os.path.exists(tree_path):
        for i in range(1, self.machine_num):
            if 2**i >= self.machine_num:
                join_max_depth = i + 1
                break
        num_dims = len(self.used_columns)
        boundary = self.table_domains
        self.partition_tree = PartitionTree(num_dims, boundary, self.sample_rate)
        self.partition_tree.name = "LJT"
        self.partition_tree.join_attr = self.used_columns.index(join_col)
        self.partition_tree.join_depth = join_max_depth
        self.partition_tree.used_columns = self.used_columns
        self.partition_tree.column_width = self.column_width
        self.partition_tree.pt_root.node_size = self.eval_len_node(self.tabledata)
        self.partition_tree.pt_root.dataset = self.tabledata
        self.partition_tree.pt_root.queryset = self.queries
        canSplit = True
        cur_depth = 0

        # 仅在join_cols不为空时执行top-layer tree construction
        while canSplit:
            canSplit = False
            leaves = self.partition_tree.get_leaves()
            cur_depth += 1

            for leaf in leaves:
                if leaf.node_size < 2 * self.block_size:
                    continue
                if cur_depth <= join_max_depth:
                    split_dim = self.partition_tree.join_attr
                    split_value = np.median(leaf.dataset[:, split_dim])
                    valid, skip, _, _ = leaf.if_split(
                        split_dim, split_value, self.block_size, self.sample_rate
                    )
                    if valid:
                        child_node1, child_node2 = self.partition_tree.apply_split(
                            leaf.nid, split_dim, split_value
                        )
                        canSplit = True
                else:
                    candidate_cuts = leaf.get_candidate_cuts(extended=True)
                    # get best candidate cut position
                    skip, max_skip, max_skip_split_dim, max_skip_split_value = (
                        0,
                        -1,
                        0,
                        0,
                    )
                    for split_dim, split_value in candidate_cuts:

                        valid, skip, _, _ = leaf.if_split(
                            split_dim, split_value, self.block_size, self.sample_rate
                        )
                        if valid and skip > max_skip:
                            max_skip = skip
                            max_skip_split_dim = split_dim
                            max_skip_split_value = split_value

                    if max_skip >= 0:
                        # if the cost become smaller, apply the cut
                        child_node1, child_node2 = self.partition_tree.apply_split(
                            leaf.nid, max_skip_split_dim, max_skip_split_value
                        )
                        canSplit = True
        end_time = time.time()
        self.partition_tree.build_time = end_time - start_time
        self.partition_tree.save_tree(tree_path)

    # AdaptDB layout
    def InitializeWithADP(self, join_depth=3):
        """
        # should I also store the candidate cut positions in Partition Node ?
        The dimension of queries should match the dimension of boundary and dataset!
        """
        join_cols = list(self.join_freqs[self.table_name].keys())
        num_dims = len(self.used_columns)
        boundary = self.table_domains
        self.partition_tree = PartitionTree(num_dims, boundary, self.sample_rate)
        self.partition_tree.name = "AdaptDB"
        self.partition_tree.join_attr = join_cols
        self.partition_tree.join_depth = join_depth
        self.partition_tree.used_columns = self.used_columns
        self.partition_tree.column_width = self.column_width
        self.partition_tree.pt_root.node_size = self.eval_len_node(self.tabledata)
        self.partition_tree.pt_root.dataset = self.tabledata
        self.partition_tree.pt_root.queryset = (
            self.queries
        )  # assume all queries overlap with the boundary
        start_time = time.time()
        self.__ADP(self.block_size, join_cols, join_depth)
        end_time = time.time()
        print(f"{self.table_name} Build Time (s):", end_time - start_time)
        self.partition_tree.build_time = end_time - start_time
        self.partition_tree.save_tree(
            f"{base_dir}/../layouts/{self.benchmark}/{self.table_name}-ADP.pkl"
        )

    def __ADP(self, block_size, join_cols, join_depth):
        print_s = True
        # top-layer tree construction
        canSplit = True
        cur_depth = 1

        # 仅在join_cols不为空时执行top-layer tree construction
        if join_cols:
            while canSplit:
                canSplit = False
                leaves = self.partition_tree.get_leaves()
                cur_depth += 1
                if cur_depth > join_depth:
                    break
                for leaf in leaves:
                    if leaf.node_size < 2 * block_size or leaf.depth < cur_depth:
                        continue
                    # 选择剩余可分配数据最大的属性作为下一个切割点
                    # temp_allocations = self.partition_tree.allocations.copy()
                    split_dim = join_cols[0]
                    split_value = np.median(leaf.dataset[:, split_dim])
                    valid, skip, _, _ = leaf.if_split(
                        split_dim, split_value, block_size, self.sample_rate
                    )
                    if valid:
                        if print_s:
                            print("AdaptDB CUT!")
                            print_s = False
                        child_node1, child_node2 = self.partition_tree.apply_split(
                            leaf.nid, split_dim, split_value
                        )
                        child_node1.depth = leaf.depth + 1
                        child_node2.depth = leaf.depth + 1
                        # self.partition_tree.allocations[split_dim] -= 2.0 / pow(2, cur_depth)
                        canSplit = True

        # bottom-layer tree construction
        CanSplit = True
        while CanSplit:
            CanSplit = False

            # for leaf in self.partition_tree.get_leaves():
            leaves = self.partition_tree.get_leaves()
            # print("# number of leaf nodes:",len(leaves))
            for leaf in leaves:

                # print("current leaf node id:",leaf.nid, "leaf node dataset size:",len(leaf.dataset))
                if leaf.node_size < 2 * block_size:
                    continue

                candidate_cuts = leaf.get_candidate_cuts()

                # get best candidate cut position
                skip, max_skip, max_skip_split_dim, max_skip_split_value = 0, -1, 0, 0
                for split_dim, split_value in candidate_cuts:

                    valid, skip, _, _ = leaf.if_split(
                        split_dim, split_value, block_size, self.sample_rate
                    )
                    if valid and skip > max_skip:
                        max_skip = skip
                        max_skip_split_dim = split_dim
                        max_skip_split_value = split_value

                if max_skip >= 0:
                    # if the cost become smaller, apply the cut
                    child_node1, child_node2 = self.partition_tree.apply_split(
                        leaf.nid, max_skip_split_dim, max_skip_split_value
                    )
                    CanSplit = True

    def InitializeWithNORA(
        self,
        saved=False,
    ):
        num_dims = len(self.used_columns)
        boundary = self.table_domains
        self.partition_tree = PartitionTree(num_dims, boundary, self.sample_rate)
        self.partition_tree.name = "NORATree"
        self.partition_tree.used_columns = self.used_columns
        self.partition_tree.column_width = self.column_width
        self.partition_tree.pt_root.node_size = self.eval_len_node(self.tabledata)
        self.partition_tree.pt_root.dataset = self.tabledata
        self.partition_tree.pt_root.queryset = (
            self.queries
        )  # assume all queries overlap with the boundary
        self.partition_tree.pt_root.generate_query_MBRs()
        start_time = time.time()
        self.__NORA(self.block_size,depth_limit=None)
        end_time = time.time()
        print(f"{self.table_name} Build Time (s):", end_time - start_time)
        self.partition_tree.build_time = end_time - start_time
        if saved:
            self.partition_tree.save_tree(
                f"{base_dir}/../layouts/{self.benchmark}/{self.table_name}-NORA.pkl"
            )


    # join tree layout
    def InitializeWithJT(
        self,
        join_depth=3,
        enable_bounding_split=False,
        saved=False,
        enable_median_extend=True,
    ):
        """
        # should I also store the candidate cut positions in Partition Node ?
        The dimension of queries should match the dimension of boundary and dataset!
        """
        join_cols = list(self.join_freqs[self.table_name].keys())
        # 1->2 2->4 3->8 4->16
        for i in range(1, self.machine_num):
            if 2**i >= self.machine_num:
                join_depth = i + 1
                break
        num_dims = len(self.used_columns)
        boundary = self.table_domains
        self.partition_tree = PartitionTree(num_dims, boundary, self.sample_rate)
        self.partition_tree.name = "JoinTree"
        self.partition_tree.join_attr = join_cols
        self.partition_tree.join_depth = join_depth
        self.partition_tree.used_columns = self.used_columns
        self.partition_tree.column_width = self.column_width
        self.partition_tree.pt_root.node_size = self.eval_len_node(self.tabledata)
        self.partition_tree.pt_root.dataset = self.tabledata
        self.partition_tree.pt_root.queryset = (
            self.queries
        )  # assume all queries overlap with the boundary
        self.partition_tree.pt_root.generate_query_MBRs()
        start_time = time.time()
        self.__JT(
            self.block_size,
            join_cols,
            join_depth,
            if_bounding_split=enable_bounding_split,
            if_median_extend=enable_median_extend,
        )
        end_time = time.time()
        print(f"{self.table_name} Build Time (s):", end_time - start_time)
        self.partition_tree.build_time = end_time - start_time
        if saved:
            self.partition_tree.save_tree(
                f"{base_dir}/../layouts/{self.benchmark}/{self.table_name}-JT.pkl"
            )

    def __JT(
        self,
        block_size,
        join_cols,
        join_depth,
        if_bounding_split=False,
        if_median_extend=True,
    ):
        print_s = True
        # top-layer tree construction
        canSplit = True
        cur_depth = 1

        bouding_active_ratio = 2

        # 仅在join_cols不为空时执行top-layer tree construction
        if not if_bounding_split and join_cols:
            while canSplit:
                canSplit = False
                leaves = self.partition_tree.get_leaves()
                cur_depth += 1
                if cur_depth > join_depth:
                    break
                for leaf in leaves:
                    if leaf.node_size < 2 * block_size or leaf.depth < cur_depth:
                        continue
                    # 选择剩余可分配数据最大的属性作为下一个切割点
                    # temp_allocations = self.partition_tree.allocations.copy()
                    split_dim = join_cols[0]
                    split_value = np.median(leaf.dataset[:, split_dim])
                    valid, skip, _, _ = leaf.if_split(
                        split_dim, split_value, block_size, self.sample_rate
                    )
                    if valid:
                        if print_s:
                            print("JoinTree CUT!")
                            print_s = False
                        child_node1, child_node2 = self.partition_tree.apply_split(
                            leaf.nid, split_dim, split_value
                        )
                        child_node1.depth = leaf.depth + 1
                        child_node2.depth = leaf.depth + 1
                        # self.partition_tree.allocations[split_dim] -= 2.0 / pow(2, cur_depth)
                        canSplit = True

        # bottom-layer tree construction
        CanSplit = True
        while CanSplit:
            CanSplit = False

            # for leaf in self.partition_tree.get_leaves():
            leaves = self.partition_tree.get_leaves()
            # print("# number of leaf nodes:",len(leaves))
            for leaf in leaves:

                # print("current leaf node id:",leaf.nid, "leaf node dataset size:",len(leaf.dataset))
                if (
                    leaf.node_size < 2 * block_size
                    or leaf.queryset is None
                    or leaf.is_irregular_shape
                    or leaf.no_valid_partition
                ):
                    continue

                if if_median_extend:

                    candidate_cuts = leaf.get_candidate_cuts()
                else:
                    candidate_cuts = leaf.get_candidate_cuts(extended=False)

                # get best candidate cut position
                (
                    skip,
                    max_skip,
                    max_skip_split_dim,
                    max_skip_split_value,
                    max_skip_split_type,
                ) = (0, -1, 0, 0, 0)

                if (
                    if_bounding_split
                    and leaf.node_size <= bouding_active_ratio * block_size
                ):
                    skip = leaf.if_general_group_split(block_size)
                    

                    if skip != False and skip > max_skip:
                        (
                            max_skip,
                            max_skip_split_dim,
                            max_skip_split_value,
                            max_skip_split_type,
                        ) = (skip, None, None, 1)
                        
                    # valid, skip = leaf.if_dual_bounding_split(split_dim, split_value, block_size, approximate = False)

                for split_dim, split_value in candidate_cuts:

                    valid, skip, left_size, right_size = leaf.if_split(
                        split_dim, split_value, block_size, self.sample_rate
                    )
                    if valid and skip > max_skip:
                        max_skip = skip
                        max_skip_split_dim = split_dim
                        max_skip_split_value = split_value

                if max_skip >= 0:
                    print("max_skip:", max_skip)
                    # if the cost become smaller, apply the cut
                    child_node1, child_node2 = self.partition_tree.apply_split(
                        leaf.nid,
                        max_skip_split_dim,
                        max_skip_split_value,
                        max_skip_split_type,
                    )
                    CanSplit = True
                else:
                    leaf.no_valid_partition = True
                    
    def __NORA(self, data_threshold, depth_limit = None):
        '''
        the general NORA algorithm, which utilize bounding split, daul-bounding split and extend candidate cuts with medians
        '''
        CanSplit = True
        while CanSplit:
            CanSplit = False           
            
            # for leaf in self.partition_tree.get_leaves():
            leaves = self.partition_tree.get_leaves()
            #print("# number of leaf nodes:",len(leaves))
            for leaf in leaves:
                
                # print("current leaf node id:",leaf.nid, "leaf node dataset size:",len(leaf.dataset))
                if leaf.node_size < 2 * data_threshold or leaf.queryset is None or (depth_limit is not None and leaf.depth >= depth_limit):
                    continue
                    
                # get best candidate cut position
                skip, max_skip, max_skip_split_dim, max_skip_split_value, max_skip_split_type = 0, -1, 0, 0, 0
                # extend the candidate cut with medians when it reach the bottom
                candidate_cuts = leaf.get_candidate_cuts(extended=True) if leaf.node_size < 4 * data_threshold else leaf.get_candidate_cuts(extended=False)     
                             
                for split_dim, split_value in candidate_cuts:

                    # first try normal split
                    valid, skip, left_size, right_size = leaf.if_split(split_dim, split_value, data_threshold)
                    if valid and skip > max_skip:
                        max_skip, max_skip_split_dim, max_skip_split_value, max_skip_split_type = skip, split_dim, split_value, 0

                    # if it's available for bounding split, try it
                    if leaf.node_size < 3 * data_threshold:
                        # try bounding split
                        valid, skip,_ = leaf.if_bounding_split(data_threshold, approximate = False)
                        if valid and skip > max_skip:
                            max_skip, max_skip_split_dim, max_skip_split_value, max_skip_split_type = skip, split_dim, split_value, 1

                    # if it's availble for dual-bounding split, try it
                    elif leaf.node_size < 4 * data_threshold and left_size < 2 * data_threshold and right_size < 2 * data_threshold:
                        # try dual-bounding split              
                        valid, skip = leaf.if_dual_bounding_split(split_dim, split_value, data_threshold, approximate = False)
                        if valid and skip > max_skip:
                            max_skip, max_skip_split_dim, max_skip_split_value, max_skip_split_type = skip, split_dim, split_value, 2

                if max_skip > 0:
                    # if the cost become smaller, apply the cut
                    child_node1, child_node2 = self.partition_tree.apply_split(leaf.nid, max_skip_split_dim, max_skip_split_value, max_skip_split_type)
                    #print(" Split on node id:", leaf.nid)
                    CanSplit = True

    def evaluate_single_table_access_cost(self):
        return self.partition_tree.evaluate_query_cost(self.queries)

    def evaluate_tree_depth(self, node, depth):
        max_depth = depth
        for nid in node.children_ids:
            max_depth = max(
                max_depth,
                self.evaluate_tree_depth(
                    self.partition_tree.nid_node_dict[nid], depth + 1
                ),
            )
        return max_depth

    def evaluate_multiple_table_access_cost(self, trees):
        group_type = 0
        if trees[list(trees.keys())[0]].name == "AdaptDB":
            group_type = 0
        elif trees[trees.keys()[0]].name == "JoinTree":
            group_type = 1
        tot_cost = {}
        for join_query in self.join_queries:  # 分别处理每条查询
            if join_query["join_relations"]:
                for join_op in join_query[
                    "join_relations"
                ]:  # 分别处理查询中的每个join操作
                    join_queryset, joined_cols, joined_trees = [], [], []
                    for join_table, join_col in join_op.items():
                        # overlapped_leaf_ids=trees[join_table].query_single(join_query['vectors'][join_table])
                        join_queryset.append(join_query["vectors"][join_table])
                        joined_cols.append(join_col)
                        joined_trees.append(trees[join_table])
                    join_eval = JoinEvaluator(
                        join_queryset, joined_cols, joined_trees
                    )  # jobv2
                    hyper_read_cost, shuffle_read_cost, hyper_cost_list = (
                        join_eval.compute_total_shuffle_hyper_cost(group_type)
                    )
                    tot_cost += hyper_read_cost + shuffle_read_cost
            else:
                for tablename in join_query["vectors"]:
                    tot_cost[tablename] = tot_cost.setdefault(tablename, 0) + trees[
                        tablename
                    ].evaluate_query_cost([join_query["vectors"][tablename]])

        return tot_cost
