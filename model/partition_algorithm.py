from concurrent.futures.thread import _worker
import numpy as np
import time
import pickle
import os
import datetime
import pandas as pd
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from model.partition_tree import PartitionTree
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
        Calculate the global sample rate uniformly based on the largest table size
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        metadata_path = f"{base_dir}/../dataset/{self.benchmark}/metadata.pkl"

        if not os.path.exists(metadata_path):
            return 1.0 

        metadata = pickle.load(open(metadata_path, "rb"))
        max_rows = 0

        for table, table_meta in metadata.items():
            rows = table_meta["rows"]
            if rows > dim_table_threshold:
                max_rows = max(max_rows, rows)

        if max_rows > max_sample_size:
            return max_sample_size / max_rows
        return 1.0

    def load_data(self, dim_table_threshold=10000):
        """
        Load data from the data source.
        """
        metadata = pickle.load(
            open(f"{base_dir}/../dataset/{self.benchmark}/metadata.pkl", "rb")
        )
        self.used_columns = metadata[self.table_name]["numeric_columns"]

        # Get table domains
        table_min_domains, table_max_domains = [], []
        for _, col_range in metadata[self.table_name]["ranges"].items():
            # Convert datetime to 'yyyy-mm-dd' format, keep other types unchanged
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

        # Read CSV data
        file_path = f"{base_dir}/../dataset/{self.benchmark}/{self.table_name}.csv"
        if self.sample_rate < 1.0 and total_rows > dim_table_threshold:
            data = pd.read_csv(
                file_path,
                usecols=self.used_columns,
                skiprows=lambda i: i > 0 and np.random.random() > self.sample_rate,
            )
            print(
                f"Sample {self.table_name}: Total rows {total_rows}, Sampled {len(data)} rows, Sample rate {self.sample_rate:.4f}"
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
        self.queries = query_vectors

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

    # AD-MTO layout
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
        canSplit = True
        cur_depth = 1
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
                        canSplit = True
        # bottom-layer tree construction
        CanSplit = True
        while CanSplit:
            CanSplit = False
            leaves = self.partition_tree.get_leaves()
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
        self.__NORA(self.block_size, depth_limit=None)
        end_time = time.time()
        print(f"{self.table_name} Build Time (s):", end_time - start_time)
        self.partition_tree.build_time = end_time - start_time
        if saved:
            self.partition_tree.save_tree(
                f"{base_dir}/../layouts/{self.benchmark}/{self.table_name}-NORA.pkl"
            )

    def cal_optimal_join_depth(self, join_cols):
        # 1->2 2->4 3->8 4->16
        try:
            best_depth,lowest_cost=0,float('inf')
            for xi in [1,2,3,4]:
                max_nodes=xi*self.machine_num
                join_depth=0
                for i in range(1, max_nodes):
                    if 2**i >= max_nodes:
                        join_depth = i + 1
                        break
                print(f"candidate join depth:{join_depth}")
                partition_tree=self.__fake_JT(self.block_size, join_cols, join_depth,self.machine_num)
                join_cost=partition_tree.estimate_join_cost(self.table_name,self.join_queries)
                scan_cost=partition_tree.evaluate_query_cost(self.queries)
                if join_cost+scan_cost<lowest_cost:
                    best_depth=join_depth
                    lowest_cost=join_cost+scan_cost
        except:
            print("Error calculating top tree depth, using default parameters instead")
            for i in range(1, max_nodes):
                if 2**i >= max_nodes:
                    best_depth = i + 1
                    break
        return best_depth

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
        join_depth = self.cal_optimal_join_depth(join_cols)
        self.partition_tree = PartitionTree(len(self.used_columns), self.table_domains, self.sample_rate)
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

    def __fake_JT(self, block_size, join_cols, join_depth, worker_num):
        partition_tree=PartitionTree(len(self.used_columns), self.table_domains, self.sample_rate)
        canSplit = True
        cur_depth = 1
        if join_cols:
            while canSplit:
                canSplit = False
                leaves = partition_tree.get_leaves()
                cur_depth += 1
                if cur_depth > join_depth:
                    break
                for leaf in leaves:
                    if leaf.node_size < 2 * block_size or leaf.depth < cur_depth:
                        continue
                    split_dim = join_cols[0]
                    split_value = np.median(leaf.dataset[:, split_dim])
                    valid, skip, _, _ = leaf.if_split(
                        split_dim, split_value, block_size, self.sample_rate
                    )
                    if valid:
                        child_node1, child_node2 = partition_tree.apply_split(
                            leaf.nid, split_dim, split_value
                        )
                        child_node1.depth = leaf.depth + 1
                        child_node2.depth = leaf.depth + 1
                        canSplit = True
            join_cost=partition_tree.estimate_join_cost(self.table_name,self.join_queries, join_cols, worker_num)
            scan_cost=partition_tree.evaluate_query_cost(self.queries)
            return join_cost+scan_cost
        else:
            return float('inf')


    def __JT(self,block_size,join_cols,join_depth,if_bounding_split=False,if_median_extend=True):
        print_s = True
        # top-layer tree construction
        canSplit = True
        cur_depth = 1

        bouding_active_ratio = 2
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

    def __NORA(self, data_threshold, depth_limit=None):
        """
        the general NORA algorithm, which utilize bounding split, daul-bounding split and extend candidate cuts with medians
        """
        CanSplit = True
        while CanSplit:
            CanSplit = False

            # for leaf in self.partition_tree.get_leaves():
            leaves = self.partition_tree.get_leaves()
            # print("# number of leaf nodes:",len(leaves))
            for leaf in leaves:

                # print("current leaf node id:",leaf.nid, "leaf node dataset size:",len(leaf.dataset))
                if (
                    leaf.node_size < 2 * data_threshold
                    or leaf.queryset is None
                    or (depth_limit is not None and leaf.depth >= depth_limit)
                ):
                    continue

                # get best candidate cut position
                (
                    skip,
                    max_skip,
                    max_skip_split_dim,
                    max_skip_split_value,
                    max_skip_split_type,
                ) = (0, -1, 0, 0, 0)
                # extend the candidate cut with medians when it reach the bottom
                candidate_cuts = (
                    leaf.get_candidate_cuts(extended=True)
                    if leaf.node_size < 4 * data_threshold
                    else leaf.get_candidate_cuts(extended=False)
                )

                for split_dim, split_value in candidate_cuts:

                    # first try normal split
                    valid, skip, left_size, right_size = leaf.if_split(
                        split_dim, split_value, data_threshold
                    )
                    if valid and skip > max_skip:
                        (
                            max_skip,
                            max_skip_split_dim,
                            max_skip_split_value,
                            max_skip_split_type,
                        ) = (skip, split_dim, split_value, 0)

                    # if it's available for bounding split, try it
                    if leaf.node_size < 3 * data_threshold:
                        # try bounding split
                        valid, skip, _ = leaf.if_bounding_split(
                            data_threshold, approximate=False
                        )
                        if valid and skip > max_skip:
                            (
                                max_skip,
                                max_skip_split_dim,
                                max_skip_split_value,
                                max_skip_split_type,
                            ) = (skip, split_dim, split_value, 1)

                    # if it's availble for dual-bounding split, try it
                    elif (
                        leaf.node_size < 4 * data_threshold
                        and left_size < 2 * data_threshold
                        and right_size < 2 * data_threshold
                    ):
                        # try dual-bounding split
                        valid, skip = leaf.if_dual_bounding_split(
                            split_dim, split_value, data_threshold, approximate=False
                        )
                        if valid and skip > max_skip:
                            (
                                max_skip,
                                max_skip_split_dim,
                                max_skip_split_value,
                                max_skip_split_type,
                            ) = (skip, split_dim, split_value, 2)

                if max_skip > 0:
                    # if the cost become smaller, apply the cut
                    child_node1, child_node2 = self.partition_tree.apply_split(
                        leaf.nid,
                        max_skip_split_dim,
                        max_skip_split_value,
                        max_skip_split_type,
                    )
                    # print(" Split on node id:", leaf.nid)
                    CanSplit = True

    def InitializeWithPando(self, max_depth=10000):
        """
        Pando Layout: Build an independent partition tree for each column, with no size limit on leaf nodes.
        When querying, scan matching blocks in each tree separately, then combine results using column-wise logical operations.
        """
        start_time = time.time()
        num_dims = len(self.used_columns)
        boundary = self.table_domains

        # Build independent trees for each column (stored in a list)
        self.pando_trees = []
        for col_idx in range(num_dims):
            tree = PartitionTree(
                1, [boundary[col_idx], boundary[num_dims + col_idx]], self.sample_rate
            )
            tree.name = f"Pando-Col{col_idx}"
            tree.used_columns = [self.used_columns[col_idx]]
            tree.column_width = {
                self.used_columns[col_idx]: self.column_width[
                    self.used_columns[col_idx]
                ]
            }
            # Keep only current column's data
            col_data = self.tabledata[:, col_idx : col_idx + 1]
            tree.pt_root.node_size = self.eval_len_node(col_data)
            tree.pt_root.dataset = col_data
            # Keep only current column's query ranges
            col_queries = []
            for q in self.queries:
                col_q = [q[col_idx], q[num_dims + col_idx]]
                col_queries.append(col_q)
            tree.pt_root.queryset = col_queries
            # Build single-column tree (no leaf size limit)
            self.__Pando_build_tree(tree, max_depth)
            self.pando_trees.append(tree)

        # Create actual physical blocks pando_blocks
        self._create_pando_blocks()

        end_time = time.time()
        print(f"{self.table_name} Pando Build Time (s):", end_time - start_time)
        self.pando_build_time = end_time - start_time

    def __Pando_build_tree(self, tree, max_depth):
        """
        Recursively build single-column partition tree, using filters from query_set as split points, with no leaf size limit.
        """
        can_split = True
        cur_depth = 1

        # Collect all filters from queries as candidate split points
        all_filters = []
        for query_range in tree.pt_root.queryset:
            lower, upper = query_range
            # Add query boundaries as potential split points
            if lower < upper:
                all_filters.extend([lower, upper])

        # Deduplicate and sort split points
        split_candidates = sorted(list(set(all_filters)))

        while can_split and cur_depth <= max_depth:
            can_split = False
            leaves = tree.get_leaves()
            cur_depth += 1

            for leaf in leaves:
                if leaf.depth >= max_depth:
                    continue

                # Get current leaf node's data range
                current_range = [leaf.boundary[0], leaf.boundary[1]]

                # Find best split point within current range
                best_split = None
                best_balance = float("inf")

                for split_value in split_candidates:
                    if current_range[0] < split_value < current_range[1]:
                        # Check if this split point effectively divides the query set
                        left_queries = 0
                        right_queries = 0

                        for query_range in leaf.queryset:
                            q_lower, q_upper = query_range
                            # Calculate query intersection with left and right subspaces (handle date strings)
                            if (
                                isinstance(split_value, str)
                                and isinstance(q_upper, str)
                                and isinstance(q_lower, str)
                            ):
                                # Date string comparison
                                left_intersect = (
                                    1
                                    if split_value >= q_lower
                                    and current_range[0] <= q_upper
                                    else 0
                                )
                                right_intersect = (
                                    1
                                    if current_range[1] >= q_lower
                                    and split_value <= q_upper
                                    else 0
                                )
                            else:
                                # Numeric comparison
                                left_intersect = max(
                                    0,
                                    min(split_value, q_upper)
                                    - max(current_range[0], q_lower),
                                )
                                right_intersect = max(
                                    0,
                                    min(current_range[1], q_upper)
                                    - max(split_value, q_lower),
                                )

                            if left_intersect > 0:
                                left_queries += 1
                            if right_intersect > 0:
                                right_queries += 1

                        # Evaluate split quality: balance left and right query counts
                        if left_queries > 0 and right_queries > 0:
                            balance = abs(left_queries - right_queries)
                            if balance < best_balance:
                                best_balance = balance
                                best_split = split_value

                # Apply best split
                if best_split is not None:
                    child_node1, child_node2 = tree.apply_split(leaf.nid, 0, best_split)
                    child_node1.depth = leaf.depth + 1
                    child_node2.depth = leaf.depth + 1
                    can_split = True

    def _create_pando_blocks(self):
        """
        Create actual physical blocks pando_blocks
        Using flattened dictionary structure with physical block index tuples as keys and tuple lists as values
        This ensures tuples are correctly assigned to corresponding physical blocks
        """
        if not hasattr(self, "pando_trees") or not self.pando_trees:
            print("Warning: pando_trees not initialized")
            return

        # Get leaf node count for each column's subtree
        col_leaf_counts = []
        for i, tree in enumerate(self.pando_trees):
            leaves = tree.get_leaves()
            col_leaf_counts.append(len(leaves))
            print(f"Debug: Column {i} subtree leaf node count: {len(leaves)}")

        print(f"Debug: Leaf node counts for all columns: {col_leaf_counts}")

        # Use flattened dictionary structure: keys are physical block index tuples, values are tuple lists
        from collections import defaultdict

        self.pando_blocks = defaultdict(list)

        # Traverse all tuples and assign to physical blocks based on routing results
        total_rows = len(self.tabledata)
        print(f"Debug: Total tuples: {total_rows}")

        assigned_tuples = 0
        for row_idx in range(total_rows):
            tuple_data = self.tabledata[row_idx]

            # Get logical block ID for this tuple in each column's subtree
            block_indices = []
            for col_idx, tree in enumerate(self.pando_trees):
                col_value = tuple_data[col_idx]

                # Find corresponding logical leaf node in this column's subtree
                leaf_id = self._find_leaf_for_value(tree, col_value)
                block_indices.append(leaf_id)

            # Assign tuple to corresponding physical block
            # Use tuple as key to ensure correct indexing
            block_key = tuple(block_indices)
            self.pando_blocks[block_key].append(tuple_data.tolist())
            assigned_tuples += 1

        # Verify assignment results
        total_blocks = len(self.pando_blocks)
        print(f"Debug: Successfully assigned {assigned_tuples} tuples to {total_blocks} physical blocks")

        # Print contents of first few blocks for verification
        block_keys = list(self.pando_blocks.keys())[:5]
        for key in block_keys:
            tuples_in_block = len(self.pando_blocks[key])
            print(f"Debug: Block {key} contains {tuples_in_block} tuples")

        # Verify total tuple count
        total_assigned = sum(len(tuples) for tuples in self.pando_blocks.values())
        print(f"Debug: Verify total assigned tuples: {total_assigned} / {total_rows}")

    def get_pando_block_count(self):
        """
        Get total number of blocks in pando_blocks
        """
        if not hasattr(self, "pando_blocks") or not self.pando_blocks:
            return 0

        # For flattened dictionary structure, directly return number of keys
        return len(self.pando_blocks)

    def get_pando_block(self, *block_indices):
        """
        Get content of specific physical block by block indices

        Args:
            *block_indices: Variable parameters, each corresponding to a column's block index

        Returns:
            List of tuples in this physical block
        """
        if not hasattr(self, "pando_blocks") or not self.pando_blocks:
            return []

        try:
            block_key = tuple(block_indices)
            return self.pando_blocks.get(block_key, [])
        except Exception:
            return []

    def _find_leaf_for_value(self, tree, value):
        """
        Find leaf node ID corresponding to given value in single column subtree
        """
        current_node = tree.pt_root

        # Traverse directly to leaf node
        while current_node.children_ids:
            if len(current_node.children_ids) == 2:
                left_child = tree.nid_node_dict[current_node.children_ids[0]]
                right_child = tree.nid_node_dict[current_node.children_ids[1]]

                # Choose child node based on boundary value
                if value <= left_child.boundary[1]:
                    current_node = left_child
                else:
                    current_node = right_child
            else:
                # Case of single child node
                current_node = tree.nid_node_dict[current_node.children_ids[0]]

        return current_node.nid


    def query_pando(self, query, logic="AND"):
        """
        Execute query on Pando layout:
        1. Query logical node IDs that meet conditions in each column's corresponding tree
        2. Find all physical blocks that need to be accessed based on Cartesian product of these IDs
        3. Count total tuples in these physical blocks

        Args:
            query: Complete query range [l1,...,ln, u1,...,un]
            logic: Logical relationship between columns, "AND" or "OR"
        Returns:
            Number of accessed tuples
        """
        if not hasattr(self, "pando_trees"):
            raise ValueError(
                "Pando layout not initialized. Call InitializeWithPando first."
            )
        if not hasattr(self, "pando_blocks"):
            raise ValueError(
                "Pando blocks not created. Call InitializeWithPando first."
            )

        # Debug info: Check pando_blocks structure
        total_blocks = self.get_pando_block_count()
        print(f"Debug: Total Pando blocks: {total_blocks}")

        num_dims = len(self.used_columns)
        col_block_lists = []

        # Query logical node IDs that meet conditions in each column's tree
        for col_idx, tree in enumerate(self.pando_trees):
            col_query = [query[col_idx], query[num_dims + col_idx]]
            col_blocks = tree.query_single(col_query)
            col_block_lists.append(col_blocks)

        # Debug info: Check block IDs returned by each column
        print(f"Debug: Logical block IDs returned by each column: {col_block_lists}")

        # Check if pando_blocks is empty
        if total_blocks == 0:
            print("Warning: pando_blocks is empty, possible creation issue")
            return 0

        # Process block lists based on logical relationship
        if logic == "AND":
            # AND logic: all columns must meet conditions
            # Check for empty results
            if any(len(blocks) == 0 for blocks in col_block_lists):
                print("Debug: At least one column has empty results, AND query returns 0")
                return 0

            # Use Cartesian product to find all possible physical block combinations
            from itertools import product

            physical_blocks = product(*col_block_lists)
            total_tuples = 0
            accessed_blocks = 0

            for block_indices in physical_blocks:
                block_content = self.get_pando_block(*block_indices)
                if block_content:  # Only count non-empty blocks
                    total_tuples += len(block_content)
                    accessed_blocks += 1

            print(f"Debug: AND logic accessed blocks: {accessed_blocks}, tuples: {total_tuples}")

        elif logic == "OR":
            # OR logic: any column meeting conditions is sufficient
            # Collect all involved tuples, avoid duplicate counting
            visited_tuples = set()

            # Get leaf node count for each column
            leaf_counts = [len(tree.get_leaves()) for tree in self.pando_trees]

            # Use set to collect all physical block indices that need to be accessed
            all_block_indices = set()

            # For each column with query conditions
            for col_idx, col_blocks in enumerate(col_block_lists):
                if col_blocks:  # This column has query results
                    # Generate all possible values for other columns
                    other_ranges = []
                    for other_col_idx in range(len(self.pando_trees)):
                        if other_col_idx == col_idx:
                            other_ranges.append(col_blocks)
                        else:
                            other_ranges.append(range(leaf_counts[other_col_idx]))

                    from itertools import product

                    # Generate all possible complete block index combinations
                    for block_indices in product(*other_ranges):
                        all_block_indices.add(block_indices)

            # Count tuples in all accessed blocks
            total_tuples = 0
            accessed_blocks = 0
            for block_indices in all_block_indices:
                block_content = self.get_pando_block(*block_indices)
                if block_content:
                    for tuple_data in block_content:
                        visited_tuples.add(tuple(tuple_data))

            total_tuples = len(visited_tuples)
            print(f"Debug: OR logic unique tuples accessed: {total_tuples}")
        else:
            raise ValueError("logic must be 'AND' or 'OR'")

        return total_tuples

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


    def evaluate_pando_cost(self, queries=None, logic="AND"):
        """
        Evaluate query cost under Pando layout, calculate data scan ratio similar to evaluate_single_table_access_cost
        Args:
            queries: Query list, use self.queries if None
            logic: Logical relationship between columns ("AND" or "OR")
        Returns:
            Average data scan ratio (ratio of accessed data volume to total data volume)
        """
        if queries is None:
            queries = self.queries
        if not hasattr(self, "pando_trees"):
            raise ValueError(
                "Pando layout not initialized. Call InitializeWithPando first."
            )

        total_scan_ratio = 0
        total_rows = len(self.tabledata)

        for count, query in enumerate(queries):
            # Use query_pando to get block IDs meeting conditions
            query_rows = self.query_pando(query, logic=logic)
            scan_ratio = query_rows / total_rows
            total_scan_ratio += scan_ratio

            print(f"Pando Query#{count}: Scan ratio: {scan_ratio:.4f}")

        avg_scan_ratio = total_scan_ratio / len(queries) if queries else 0
        print(f"Pando average data scan ratio (logic={logic}): {avg_scan_ratio:.4f}")
        return avg_scan_ratio
