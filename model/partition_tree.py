import time

import pickle
from model.partition_node import PartitionNode
import numpy as np
from numpy import genfromtxt
import copy
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D


class PartitionTree:
    """
    The data structure that represent the partition layout, which also maintain the parent, children relation info
    Designed to provide efficient online query and serialized ability

    The node data structure could be checked from the PartitionNode class

    """

    def __init__(self, num_dims=0, boundary=[], sample_rate=1.0):

        # the node id of root should be 0, its pid should be -1
        # note this initialization does not need dataset and does not set node size!

        self.pt_root = PartitionNode(
            num_dims,
            boundary,
            nid=0,
            pid=-1,
            is_irregular_shape_parent=False,
            is_irregular_shape=False,
            num_children=0,
            children_ids=[],
            is_leaf=True,
            node_size=0,
        )
        self.nid_node_dict = {0: self.pt_root}  # node id to node dictionary
        self.node_count = 1  # the root node
        self.sample_rate = sample_rate
        # self.allocations=list()
        # self.table_dataset_size=None # record the table data size

    # = = = = = public functions (API) = = = = =

    def save_tree(self, path):
        node_list = self.__generate_node_list(
            self.pt_root
        )  # do we really need this step?
        serialized_node_list = self.__serialize_by_pickle(node_list)
        with open(path, "wb") as f:
            # perform different operations for join tree and normal tree
            if hasattr(self, "join_attr"):
                pickle.dump(
                    (
                        self.join_attr,
                        self.join_depth,
                        self.column_width,
                        self.used_columns,
                        self.sample_rate,
                        serialized_node_list,
                    ),
                    f,
                    True,
                )
            else:
                pickle.dump(
                    (
                        self.column_width,
                        self.used_columns,
                        self.sample_rate,
                        serialized_node_list,
                    ),
                    f,
                    True,
                )
        # return serialized_node_list

    def load_tree(self, path):
        with open(path, "rb") as f:
            res = pickle.load(f)
            if hasattr(self, "join_attr"):
                self.join_attr = res[0]
                self.join_depth = res[1]
                self.column_width = res[2]
                self.used_columns = res[3]
                self.sample_rate = res[4]
                serialized_node_list = res[5]
            else:
                self.column_width = res[0]
                self.used_columns = res[1]
                self.sample_rate = res[2]
                serialized_node_list = res[3]
        self.__build_tree_from_serialized_node_list(serialized_node_list)

        self.node_count = len(self.nid_node_dict.keys())

    # 仅仅访问分区树top layer的节点，确定查询的大范围（即join key的范围）
    def query_single_toplayer(self, query):
        partition_ids = self.__find_overlapped_partition_consider_depth(
            self.pt_root, query, 1
        )
        return partition_ids

    def query_single(self, query, print_info=False):
        """
        query is in plain form, i.e., [l1,l2,...,ln, u1,u2,...,un]
        return the overlapped leaf partitions ids!
        redundant_partition: [(boundary, size)...]
        """

        # used only when redundant_partition is given
        def check_inside(query, partition_boundary):
            num_dims = len(query) // 2
            for i in range(num_dims):
                if (
                    query[i] >= partition_boundary[i]
                    and query[num_dims + i] <= partition_boundary[num_dims + i]
                ):
                    pass
                else:
                    return False
            return True

        block_ids = self.__find_overlapped_partition(self.pt_root, query, print_info)
        return block_ids

    def get_queryset_cost(self, queries):
        """
        return the cost array directly
        """
        costs = []
        for query in queries:
            overlapped_leaf_ids = self.query_single(query)
            cost = 0
            for nid in overlapped_leaf_ids:
                cost += self.nid_node_dict[nid].node_size
            costs.append(cost)
        return costs

    def evaluate_query_cost(self, queries, print_result=False):
        # if len(queries)==0: return 0
        """
        get the logical IOs of the queris
        return the average query cost
        """
        total_cost = 0
        total_ratio = 0
        case = 0
        total_overlap_ids = {}
        case_cost = {}

        tablesize = self.pt_root.node_size
        column_width = self.column_width
        accessed_cols = self.used_columns

        row_width = sum([column_width[col] for col in column_width])
        access_width = sum([column_width[col] for col in accessed_cols])

        for count, query in enumerate(queries):
            cost = 0
            overlapped_leaf_ids = self.query_single(query)
            print(f"~~~~~~~~~Query#{count}: {overlapped_leaf_ids}")

            total_overlap_ids[case] = overlapped_leaf_ids
            actual_data_size = []
            for nid in overlapped_leaf_ids:
                cur_node = self.nid_node_dict[nid]
                cost += cur_node.node_size
                actual_data_size.append(cur_node.node_size)

            print(
                f"query #{count}:  Row Count: {sum(actual_data_size)}, Access Ratio:{sum(actual_data_size)*access_width/(tablesize*row_width)}"
            )
            total_ratio += (
                sum(actual_data_size) * access_width / (tablesize * row_width)
            )
            total_cost += cost
            case_cost[case] = cost
            case += 1

        print("Total logical IOs:", total_cost)
        print("Average logical IOs:", total_cost // len(queries))

        return total_ratio / len(queries)

    def get_pid_for_data_point(self, point):
        """
        get the corresponding leaf partition nid for a data point
        point: [dim1_value, dim2_value...], contains the same dimenions as the partition tree
        """
        return self.__find_resided_partition(self.pt_root, point)

    def add_node(self, parent_id, child_node):
        child_node.nid = self.node_count
        self.node_count += 1

        child_node.pid = parent_id
        self.nid_node_dict[child_node.nid] = child_node

        child_node.depth = self.nid_node_dict[parent_id].depth + 1

        self.nid_node_dict[parent_id].children_ids.append(child_node.nid)
        self.nid_node_dict[parent_id].num_children += 1
        self.nid_node_dict[parent_id].is_leaf = False

    def eval_len_node(self, dataset):
        return int(len(dataset) // self.sample_rate)

    def apply_split(
        self,
        parent_nid,
        split_dim,
        split_value,
        split_type=0,
        extended_bound=None,
        approximate=False,
        pretend=False,
    ):
        """
        split_type = 0: split a node into 2 sub-nodes by a given dimension and value, distribute dataset
        split_type = 1: split a node by bounding split (will create an irregular shape partition)
        split_type = 2: split a node by daul-bounding split (will create an irregular shape partition)
        split_type = 3: split a node by var-bounding split (multi MBRs), distribute dataset
        extended_bound is only used in split type 1
        approximate: used for measure query result size
        pretend: if pretend is True, return the split result, but do not apply this split
        """
        parent_node = self.nid_node_dict[parent_nid]
        if pretend:
            parent_node = copy.deepcopy(self.nid_node_dict[parent_nid])

        child_node1, child_node2 = None, None

        def special_copy(target_node):
            new_node = copy.copy(target_node)
            new_node.boundary = copy.deepcopy(target_node.boundary)
            # new_node.children_ids = copy.deepcopy(target_node.children_ids)
            new_node.dataset = None
            new_node.queryset = None
            return new_node

        if split_type == 0:  # single-step split

            # create sub nodes
            child_node1 = special_copy(parent_node)
            child_node1.boundary[split_dim + child_node1.num_dims] = split_value
            child_node1.children_ids = []

            child_node2 = special_copy(parent_node)
            child_node2.boundary[split_dim] = split_value
            child_node2.children_ids = []

            # if parent_node.dataset != None: # The truth value of an array with more than one element is ambiguous.
            # https://stackoverflow.com/questions/36783921/valueerror-when-checking-if-variable-is-none-or-numpy-array
            if parent_node.dataset is not None:
                child_node1.dataset = parent_node.dataset[
                    parent_node.dataset[:, split_dim] < split_value
                ]
                child_node1.node_size = self.eval_len_node(child_node1.dataset)
                child_node2.dataset = parent_node.dataset[
                    parent_node.dataset[:, split_dim] >= split_value
                ]
                child_node2.node_size = self.eval_len_node(child_node2.dataset)

            if parent_node.queryset is not None:
                left_part, right_part, mid_part = parent_node.split_queryset(
                    split_dim, split_value
                )
                child_node1.queryset = left_part + mid_part
                child_node2.queryset = right_part + mid_part

            if parent_node.query_MBRs is not None:
                MBRs1, MBRs2 = parent_node.split_query_MBRs(split_dim, split_value)
                child_node1.query_MBRs = MBRs1
                child_node2.query_MBRs = MBRs2

            # update current node
            if not pretend:
                self.add_node(parent_nid, child_node1)
                self.add_node(parent_nid, child_node2)
                self.nid_node_dict[parent_nid].split_type = "candidate cut"
        elif split_type == 1:  # bound split
            remaining_size = parent_node.node_size
            for MBR in parent_node.query_MBRs:
                child_node = special_copy(parent_node)
                child_node.is_leaf = True
                child_node.children_ids = []
                child_node.boundary = MBR.boundary
                child_node.node_size = MBR.bound_size
                child_node.partitionable = False
                remaining_size -= child_node.node_size
                child_node.dataset = self.__extract_sub_dataset(parent_node.dataset, child_node.boundary)
                child_node.queryset = MBR.queries # no other queries could overlap this MBR, or it's invalid
                child_node.query_MBRs = [MBR]
                if not pretend:
                    self.add_node(parent_nid, child_node)

            # the last irregular shape partition
            child_node = special_copy(parent_node)
            child_node.is_leaf = True
            child_node.children_ids = []
            child_node.is_irregular_shape = True
            child_node.node_size = remaining_size
            child_node.partitionable = False
            child_node.dataset = None
            child_node.queryset = None
            child_node.query_MBRs = None
            if not pretend:
                self.add_node(parent_nid, child_node)
                self.nid_node_dict[parent_nid].is_irregular_shape_parent = True
                self.nid_node_dict[parent_nid].split_type = "var-bounding split"
        else:
            print("Invalid Split Type!")
        # real split
        if not pretend:
            del self.nid_node_dict[parent_nid].dataset
            del self.nid_node_dict[parent_nid].queryset

        return child_node1, child_node2

    def get_leaves(self, use_partitionable=False):
        nodes = []
        if use_partitionable:
            for nid, node in self.nid_node_dict.items():
                if node.is_leaf and node.partitionable:
                    nodes.append(node)
        else:
            for nid, node in self.nid_node_dict.items():
                if node.is_leaf:
                    nodes.append(node)
        return nodes

    def visualize(
        self,
        dims=[0, 1],
        queries=[],
        path=None,
        focus_region=None,
        add_text=True,
        use_sci=False,
    ):
        """
        visualize the partition tree's leaf nodes
        focus_region: in the shape of boundary
        """
        if len(dims) == 2:
            self.__visualize_2d(dims, queries, path, focus_region, add_text, use_sci)
        else:
            self.__visualize_3d(dims[0:3], queries, path, focus_region)

    # = = = = = internal functions = = = = =

    def __extract_sub_dataset(self, dataset, query):
        constraints = []
        num_dims = self.pt_root.num_dims
        for d in range(num_dims):
            constraint_L = dataset[:, d] >= query[d]
            constraint_U = dataset[:, d] <= query[num_dims + d]
            constraints.append(constraint_L)
            constraints.append(constraint_U)
        constraint = np.all(constraints, axis=0)
        sub_dataset = dataset[constraint]
        return sub_dataset

    def __generate_node_list(self, node):
        """
        recursively add childrens into the list
        """
        node_list = [node]
        for nid in node.children_ids:
            node_list += self.__generate_node_list(self.nid_node_dict[nid])
        return node_list

    def __serialize_by_pickle(self, node_list):
        serialized_node_list = []
        for node in node_list:
            attributes = [node.num_dims, node.boundary, node.nid, node.pid]
            attributes.append(1 if node.is_irregular_shape_parent else 0)
            attributes.append(1 if node.is_irregular_shape else 0)
            attributes.append(node.num_children)
            attributes.append(1 if node.is_leaf else 0)
            attributes.append(node.node_size)
            attributes.append(node.depth)
            serialized_node_list.append(attributes)
        return serialized_node_list

    def __build_tree_from_serialized_node_list(self, serialized_node_list):

        self.pt_root = None
        self.nid_node_dict.clear()
        pid_children_ids_dict = {}
        for serialized_node in serialized_node_list:
            num_dims = serialized_node[0]
            boundary = serialized_node[1]
            nid = serialized_node[2]
            pid = serialized_node[3]
            is_irregular_shape_parent = False if serialized_node[4] == 0 else True
            is_irregular_shape = False if serialized_node[5] == 0 else True
            num_children = serialized_node[6]
            is_leaf = False if serialized_node[7] == 0 else True
            node_size = serialized_node[8]
            node = PartitionNode(
                num_dims,
                boundary,
                nid,
                pid,
                is_irregular_shape_parent,
                is_irregular_shape,
                num_children,
                [],
                is_leaf,
                node_size,
            )
            node.depth = serialized_node[9]
            self.nid_node_dict[nid] = node  # update dict

            if node.pid in pid_children_ids_dict:
                pid_children_ids_dict[node.pid].append(node.nid)
            else:
                pid_children_ids_dict[node.pid] = [node.nid]

        # make sure the irregular shape partition is placed at the end of the child list
        for pid, children_ids in pid_children_ids_dict.items():
            if pid == -1:
                continue
            if (
                self.nid_node_dict[pid].is_irregular_shape_parent
                and not self.nid_node_dict[children_ids[-1]].is_irregular_shape
            ):
                # search for the irregular shape partition
                new_children_ids = []
                irregular_shape_id = None
                for nid in children_ids:
                    if self.nid_node_dict[nid].is_irregular_shape:
                        irregular_shape_id = nid
                    else:
                        new_children_ids.append(nid)
                new_children_ids.append(irregular_shape_id)
                self.nid_node_dict[pid].children_ids = new_children_ids
            else:
                self.nid_node_dict[pid].children_ids = children_ids

        self.pt_root = self.nid_node_dict[0]

    def __bound_query_by_boundary(self, query, boundary):
        """
        bound the query by a node's boundary
        """
        bounded_query = query.copy()
        num_dims = self.pt_root.num_dims
        for dim in range(num_dims):
            bounded_query[dim] = max(query[dim], boundary[dim])
            bounded_query[num_dims + dim] = min(
                query[num_dims + dim], boundary[num_dims + dim]
            )
        return bounded_query

    def __find_resided_partition(self, node, point):
        """
        for data point only
        """
        # print("enter function!")
        if node.is_leaf:
            # print("within leaf",node.nid)
            if node.is_contain(point):
                if node.linked_ids:
                    write_ids = [node.nid]
                    for link_id in node.linked_ids:
                        link_node = self.nid_node_dict[link_id]
                        if link_node.is_redundant_contain(point):
                            write_ids.append(link_id)
                    return write_ids
                return [node.nid]

        for nid in node.children_ids:
            if self.nid_node_dict[nid].is_contain(point):
                # print("within child", nid, "of parent",node.nid)
                return self.__find_resided_partition(self.nid_node_dict[nid], point)

        # print("no children of node",node.nid,"contains point")
        return [-1]

    def __find_overlapped_partition_consider_depth(self, node, query, depth):
        if node.is_leaf or depth == self.join_depth:
            return [node.nid] if node.is_overlap(query) > 0 else []
        else:
            node_id_list = []
            for nid in node.children_ids:
                node_id_list += self.__find_overlapped_partition_consider_depth(
                    self.nid_node_dict[nid], query, depth + 1
                )
            return node_id_list

    def __find_overlapped_partition(self, node, query, print_info=False):

        if print_info:
            print("Enter node", node.nid)

        if node.is_leaf:
            if print_info:
                print("node", node.nid, "is leaf")

            if print_info and node.is_overlap(query) > 0:
                print("node", node.nid, "is added as result")

            if node.is_overlap(query) > 0:
                return [node.nid]
            else:
                return []

        node_id_list = []
        if node.is_overlap(query) > 0:
            if print_info:
                print("searching childrens for node", node.nid)
            for nid in node.children_ids:
                node_id_list += self.__find_overlapped_partition(
                    self.nid_node_dict[nid], query, print_info
                )
        return list(set(node_id_list))

    def __visualize_2d(
        self,
        dims,
        queries=[],
        path=None,
        focus_region=None,
        add_text=True,
        use_sci=False,
    ):
        """
        focus_region: in the shape of boundary
        """
        fig, ax = plt.subplots(1)
        num_dims = self.pt_root.num_dims

        plt.xlim(
            self.pt_root.boundary[dims[0]], self.pt_root.boundary[dims[0] + num_dims]
        )
        plt.ylim(
            self.pt_root.boundary[dims[1]], self.pt_root.boundary[dims[1] + num_dims]
        )

        case = 0
        for query in queries:
            lower1 = query[dims[0]]
            lower2 = query[dims[1]]
            upper1 = query[dims[0] + num_dims]
            upper2 = query[dims[1] + num_dims]

            rect = Rectangle(
                (lower1, lower2),
                upper1 - lower1,
                upper2 - lower2,
                fill=False,
                edgecolor="r",
                linewidth=1,
            )
            if add_text:
                ax.text(upper1, upper2, case, color="b", fontsize=7)
            case += 1
            ax.add_patch(rect)

        leaves = self.get_leaves()
        for leaf in leaves:
            lower1 = leaf.boundary[dims[0]]
            lower2 = leaf.boundary[dims[1]]
            upper1 = leaf.boundary[dims[0] + num_dims]
            upper2 = leaf.boundary[dims[1] + num_dims]

            rect = Rectangle(
                (lower1, lower2),
                upper1 - lower1,
                upper2 - lower2,
                fill=False,
                edgecolor="black",
                linewidth=1,
            )
            if add_text:
                ax.text(lower1, lower2, leaf.nid, fontsize=7)
            ax.add_patch(rect)
        # if self.pt_root.query_MBRs:
        #     for MBR in self.pt_root.query_MBRs:
        #         lower1 = MBR.boundary[dims[0]]
        #         lower2 = MBR.boundary[dims[1]]
        #         upper1 = MBR.boundary[dims[0]+num_dims]
        #         upper2 = MBR.boundary[dims[1]+num_dims]
        #
        #         rect = Rectangle((lower1,lower2),upper1-lower1,upper2-lower2,fill=False,edgecolor='y',linewidth=1)
        #         ax.add_patch(rect)

        ax.set_xlabel("dimension 1", fontsize=15)
        ax.set_ylabel("dimension 2", fontsize=15)
        if use_sci:
            plt.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
        # plt.xticks(np.arange(0, 400001, 100000), fontsize=10)
        # plt.yticks(np.arange(0, 20001, 5000), fontsize=10)

        plt.tight_layout()  # preventing clipping the labels when save to pdf
        if focus_region is not None:

            # reform focus region into interleaf format
            formated_focus_region = []
            for i in range(2):
                formated_focus_region.append(focus_region[i])
                formated_focus_region.append(focus_region[2 + i])

            plt.axis(formated_focus_region)
        path = f"/home/liupengju/pycharmProjects/NORA_JOIN_SIMULATION/images/{self.name}.png"
        if path is not None:
            fig.savefig(path)
        plt.show()

    # %matplotlib notebook
    def __visualize_3d(self, dims, queries=[], path=None, focus_region=None):
        fig = plt.figure()
        ax = Axes3D(fig)

        num_dims = self.pt_root.num_dims
        plt.xlim(
            self.pt_root.boundary[dims[0]], self.pt_root.boundary[dims[0] + num_dims]
        )
        plt.ylim(
            self.pt_root.boundary[dims[1]], self.pt_root.boundary[dims[1] + num_dims]
        )
        ax.set_zlim(
            self.pt_root.boundary[dims[2]], self.pt_root.boundary[dims[2] + num_dims]
        )

        leaves = self.get_leaves()
        for leaf in leaves:

            L1 = leaf.boundary[dims[0]]
            L2 = leaf.boundary[dims[1]]
            L3 = leaf.boundary[dims[2]]
            U1 = leaf.boundary[dims[0] + num_dims]
            U2 = leaf.boundary[dims[1] + num_dims]
            U3 = leaf.boundary[dims[2] + num_dims]

            # the 12 lines to form a rectangle
            x = [L1, U1]
            y = [L2, L2]
            z = [L3, L3]
            ax.plot3D(x, y, z, color="g")
            y = [U2, U2]
            ax.plot3D(x, y, z, color="g")
            z = [U3, U3]
            ax.plot3D(x, y, z, color="g")
            y = [L2, L2]
            ax.plot3D(x, y, z, color="g")

            x = [L1, L1]
            y = [L2, U2]
            z = [L3, L3]
            ax.plot3D(x, y, z, color="g")
            x = [U1, U1]
            ax.plot3D(x, y, z, color="g")
            z = [U3, U3]
            ax.plot3D(x, y, z, color="g")
            x = [L1, L1]
            ax.plot3D(x, y, z, color="g")

            x = [L1, L1]
            y = [L2, L2]
            z = [L3, U3]
            ax.plot3D(x, y, z, color="g")
            x = [U1, U1]
            ax.plot3D(x, y, z, color="g")
            y = [U2, U2]
            ax.plot3D(x, y, z, color="g")
            x = [L1, L1]
            ax.plot3D(x, y, z, color="g")

        for query in queries:

            L1 = query[dims[0]]
            L2 = query[dims[1]]
            L3 = query[dims[2]]
            U1 = query[dims[0] + num_dims]
            U2 = query[dims[1] + num_dims]
            U3 = query[dims[2] + num_dims]

            # the 12 lines to form a rectangle
            x = [L1, U1]
            y = [L2, L2]
            z = [L3, L3]
            ax.plot3D(x, y, z, color="r")
            y = [U2, U2]
            ax.plot3D(x, y, z, color="r")
            z = [U3, U3]
            ax.plot3D(x, y, z, color="r")
            y = [L2, L2]
            ax.plot3D(x, y, z, color="r")

            x = [L1, L1]
            y = [L2, U2]
            z = [L3, L3]
            ax.plot3D(x, y, z, color="r")
            x = [U1, U1]
            ax.plot3D(x, y, z, color="r")
            z = [U3, U3]
            ax.plot3D(x, y, z, color="r")
            x = [L1, L1]
            ax.plot3D(x, y, z, color="r")

            x = [L1, L1]
            y = [L2, L2]
            z = [L3, U3]
            ax.plot3D(x, y, z, color="r")
            x = [U1, U1]
            ax.plot3D(x, y, z, color="r")
            y = [U2, U2]
            ax.plot3D(x, y, z, color="r")
            x = [L1, L1]
            ax.plot3D(x, y, z, color="r")
        # path=f'/home/liupengju/pycharmProjects/NORA_JOIN_SIMULATION/images/3d/{time.time()}.png'
        if path is not None:
            fig.savefig(path)

        plt.show()
