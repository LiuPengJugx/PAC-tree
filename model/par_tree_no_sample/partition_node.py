import numpy as np
import copy

"""
A node class used to construct partition tree.
"""
class PartitionNode:
    '''
    A partition node, including both the internal and leaf nodes in the partition tree
    '''

    def __init__(self, num_dims=0, boundary=[], nid=None,
                 pid=None, is_irregular_shape_parent=False,
                 is_irregular_shape=False, num_children=0, children_ids=[], is_leaf=True, node_size=0):

        # print("Initialize PartitionTree Root: num_dims",num_dims,"boundary:",boundary,"children_ids:",children_ids)
        self.num_dims = num_dims  # number of dimensions
        # the domain, [l1,l2,..,ln, u1,u2,..,un,], for irregular shape partition, one need to exempt its siblings
        self.boundary = boundary  # I think the lower side should be inclusive and the upper side should be exclusive?
        self.nid = nid  # node id
        self.pid = pid  # parent id
        self.is_irregular_shape_parent = is_irregular_shape_parent  # whether the [last] child is an irregular shape partition
        self.is_irregular_shape = is_irregular_shape  # an irregular shape partition cannot be further split, and it must be a leaf node
        self.num_children = num_children  # number of children, should be 0, 2, or 3
        self.children_ids = children_ids  # if it's the irregular shape parent, then the last child should be the irregular partition
        self.is_leaf = is_leaf
        self.node_size = node_size  # number of records in this partition

        # the following attributes will not be serialized
        self.dataset = None  # only used in partition algorithms, temporary, should consist records that within this partition
        self.queryset = None  # only used in partition algorithms, temporary, should consist queries that overlap this partition
        
        # beam search
        self.depth = 0  # only used in beam search, root node depth is 0

    def is_overlap(self, query):
        '''
        query is in plain form, i.e., [l1,l2,...,ln, u1,u2,...,un]
        !query dimension should match the partition dimensions! i.e., all projected or all not projected
        return 0 if no overlap
        return 1 if overlap
        return 2 if inside
        '''
        if len(query) != 2 * self.num_dims:
            return -1  # error

        overlap_flag = True
        inside_flag = True

        for i in range(self.num_dims):
            if query[i] > self.boundary[self.num_dims + i] or query[self.num_dims + i] < self.boundary[i]:
                overlap_flag = False
                inside_flag = False
                return 0
            elif query[i] < self.boundary[i] or query[self.num_dims + i] > self.boundary[self.num_dims + i]:
                inside_flag = False

        if inside_flag:
            return 2
        elif overlap_flag:
            return 1
        else:
            return 0

    def is_overlap_np(self, query):
        '''
        the numpy version of the is_overlap function
        the query here and boundary class attribute should in the form of numpy array
        '''
        if all((self.boundary[0:self.num_dims] > query[self.num_dims:]) | (
                self.boundary[self.num_dims:] <= query[0:self.num_dims])):
            return 0  # no overlap
        elif all((self.boundary[0:self.num_dims] >= query[0:self.num_dims]) & (
                self.boundary[self.num_dims:] <= query[self.num_dims:])):
            return 2  # inside
        else:
            return 1  # overlap

    def is_redundant_contain(self, point):
        '''
        used to determine wheter a data point is contained in this node
        point: [dim1_value, dim2_value,...], should has the same dimensions as this node
        '''
        flag=False
        for boundary in self.redundant_boundaries:
            is_located=True
            for i in range(self.num_dims):
                if point[i] > boundary[self.num_dims + i] or point[i] < boundary[i]:
                    is_located=False
                    break
            if not is_located:
                continue
            else:
                flag=True
                break
        return flag
    def is_contain(self, point):
        '''
        used to determine wheter a data point is contained in this node
        point: [dim1_value, dim2_value,...], should has the same dimensions as this node
        '''
        for i in range(self.num_dims):
            if point[i] > self.boundary[self.num_dims + i] or point[i] < self.boundary[i]:
                return False
        return True


    def get_candidate_cuts(self, extended=True, begin_pos=0):
        '''
        get the candidate cut positions
        if extended is set to True, also add medians from all dimensions
        '''
        candidate_cut_pos = []
        for query in self.queryset:
            for dim in range(begin_pos,self.num_dims):
                # check if the cut position is inside the partition, as the queryset are queries overlap this partition
                if query[dim] > self.boundary[dim] and query[dim] < self.boundary[self.num_dims + dim]:
                    candidate_cut_pos.append((dim, query[dim]))
                if query[self.num_dims + dim] > self.boundary[dim] and query[self.num_dims + dim] < self.boundary[
                    self.num_dims + dim]:
                    candidate_cut_pos.append((dim, query[self.num_dims + dim]))

        if extended:
            for dim in range(self.num_dims):
                try:
                    split_value = float(np.median(self.dataset[:, dim]))
                    candidate_cut_pos.append((dim, split_value))
                except Exception as e:
                    pass
        
        return candidate_cut_pos

    def get_candidate_join_cuts(self,join_attr):
        dim=join_attr
        candidate_cut_pos = []
        for query in self.queryset:
            if query[dim] >= self.boundary[dim] and query[dim] <= self.boundary[self.num_dims + dim]:
                candidate_cut_pos.append((dim, query[dim]))
            if query[self.num_dims + dim] >= self.boundary[dim] and query[self.num_dims + dim] <= self.boundary[self.num_dims + dim]:
                candidate_cut_pos.append((dim, query[self.num_dims + dim]))
        split_value = np.median(self.dataset[:, dim])
        candidate_cut_pos.append((dim, split_value))
        return candidate_cut_pos

    def if_split(self, split_dim, split_value, data_threshold, sample_rate=1, test=False):  # rename: if_split_get_gain
        '''
        return the skip gain and children partition size if split a node from a given split dimension and split value
        '''
        # print("current_node.nid:", current_node.nid)
        # print("current_node.is_leaf:", current_node.is_leaf)
        # print("current_node.dataset is None:", current_node.dataset is None)
        sub_dataset1_size = int(np.count_nonzero(self.dataset[:, split_dim] < split_value)//sample_rate)  # process time: 0.007
        sub_dataset2_size = self.node_size - sub_dataset1_size

        if sub_dataset1_size < data_threshold or sub_dataset2_size < data_threshold:
            return False, 0, sub_dataset1_size, sub_dataset2_size

        left_part, right_part, mid_part = self.split_queryset(split_dim, split_value)
        num_overlap_child1 = len(left_part) + len(mid_part)
        num_overlap_child2 = len(right_part) + len(mid_part)

        if test:
            print("num left part:", len(left_part), "num right part:", len(right_part), "num mid part:", len(mid_part))
            print("left part:", left_part, "right part:", right_part, "mid part:", mid_part)

        # temp_child_node1, temp_child_node2 = self.__if_split_get_child(split_dim, split_value)
        skip_gain = len(
            self.queryset) * self.node_size - num_overlap_child1 * sub_dataset1_size - num_overlap_child2 * sub_dataset2_size
        return True, skip_gain, sub_dataset1_size, sub_dataset2_size


    def num_query_crossed(self, split_dim, split_value):
        '''
        similar to the split_queryset function, but just return how many queries the intended split will cross
        '''
        count = 0
        if self.queryset is not None:
            for query in self.queryset:
                if query[split_dim] < split_value and query[self.num_dims + split_dim] > split_value:
                    count += 1
            return count
        return None

    def split_queryset(self, split_dim, split_value):
        '''
        split the queryset into 3 parts:
        the left part, the right part, and those cross the split value
        '''
        if self.queryset is not None:
            left_part = []
            right_part = []
            mid_part = []
            for query in self.queryset:
                # print("[Split Queryset] query:",query, "split dim:", split_dim, "split value", split_value, "query[split dim]:",query[split_dim])
                if query[split_dim] >= split_value:
                    # print("[Split Queryset] query is right part")
                    right_part.append(query)
                elif query[self.num_dims + split_dim] <= split_value:
                    # print("[Split Queryset] query is left part")
                    left_part.append(query)
                elif query[split_dim] < split_value and query[self.num_dims + split_dim] > split_value:
                    # print("[Split Queryset] query is mid part")
                    mid_part.append(query)
                else:
                    # print("[Split Queryset] query is nothing")
                    pass
            # print("[Split Queryset] left part:",len(left_part), "right part:",len(right_part),"mid part:",len(mid_part))
            return left_part, right_part, mid_part
    def get_query_result(self,query):
        constraints = []
        for d in range(self.num_dims):
            constraint_L = self.dataset[:, d] >= query[d]
            constraint_U = self.dataset[:, d] <= query[self.num_dims + d]
            constraints.append(constraint_L)
            constraints.append(constraint_U)
        constraint = np.all(constraints, axis=0)
        return self.dataset[np.argwhere(constraint==True).flatten()]

    def query_result_size(self, query, approximate=False):
        '''
        get the query result's size on this node
        the approximate parameter is set to True, the use even distribution to approximate
        '''
        if query is None:
            return None

        result_size = 0
        if approximate:
            query_volume = 1
            volume = 1
            for d in range(self.num_dims):
                query_volume *= query[self.num_dims + d] - query[d]
                volume *= self.boundary[self.num_dims + d] - self.boundary[d]

            result_size = int(query_volume / volume * self.node_size)
        else:
            constraints = []
            for d in range(self.num_dims):
                constraint_L = self.dataset[:, d] >= query[d]
                constraint_U = self.dataset[:, d] <= query[self.num_dims + d]
                constraints.append(constraint_L)
                constraints.append(constraint_U)
            constraint = np.all(constraints, axis=0)
            result_size = np.count_nonzero(constraint)
        return result_size

    

    def extend_bound(self, bound, data_threshold, print_info=False, algorithm=2):
        '''
        extend a bound to be at least b, assume the bound is within the partition boundary
        algorithm == 1: binary search on each dimension
        algorithm == 2: Ken's extend bound method
        '''
        # safe guard
        current_size = self.query_result_size(bound, approximate=False)
        if current_size >= data_threshold:
            return bound, current_size

        if algorithm == 1:
            side = 0
            for dim in range(
                    self.num_dims):  # or it cannot adapted to other dataset ! #[2,0,1,4,3,5,6]: reranged by distinct values
                if dim + 1 > self.num_dims:
                    continue
                valid, bound, bound_size = self.__try_extend(bound, dim, 0, data_threshold, print_info)  # lower side
                if print_info:
                    print("dim:", dim, "current bound:", bound, valid, bound_size)
                if valid:
                    break
                valid, bound, bound_size = self.__try_extend(bound, dim, 1, data_threshold, print_info)  # upper side
                if print_info:
                    print("dim:", dim, "current bound:", bound, valid, bound_size)
                if valid:
                    break
            return bound, bound_size

        elif algorithm == 2:
            center = [(bound[i] + bound[i + self.num_dims]) / 2 for i in range(self.num_dims)]
            radius = [(bound[i + self.num_dims] - bound[i]) / 2 for i in range(self.num_dims)]
            f_records = []
            for point in self.dataset:
                dist_ratio = [abs(point[i] - center[i]) / radius[i] for i in range(self.num_dims)]
                max_dist_ratio = max(dist_ratio)
                f_records.append(max_dist_ratio)
            f_records.sort()
            threshold_ratio = f_records[data_threshold]
            extend_bound_lower = [center[i] - threshold_ratio * radius[i] for i in range(self.num_dims)]
            extend_bound_upper = [center[i] + threshold_ratio * radius[i] for i in range(self.num_dims)]
            extended_bound = extend_bound_lower + extend_bound_upper
            extended_bound = self.__max_bound_single(extended_bound)
            bound_size = self.query_result_size(extended_bound, approximate=False)
            return extended_bound, bound_size


    # = = = = = internal functions = = = = =

    def __try_extend(self, current_bound, try_dim, side, data_threshold, print_info=False):
        '''
        side = 0: lower side
        side = 1: upper side
        return whether this extend has made bound greater than b, current extended bound, and the size
        '''
        # first try the extreme case
        dim = try_dim
        if side == 1:
            dim += self.num_dims

        extended_bound = copy.deepcopy(current_bound)
        extended_bound[dim] = self.boundary[dim]

        bound_size = self.query_result_size(extended_bound, approximate=False)
        if bound_size < data_threshold:
            return False, extended_bound, bound_size

        # binary search in this extend direction
        L, U = None, None
        if side == 0:
            L, U = self.boundary[dim], current_bound[dim]
        else:
            L, U = current_bound[dim], self.boundary[dim]

        if print_info:
            print("L,U:", L, U)

        loop_count = 0
        while L < U and loop_count < 30:
            mid = (L + U) / 2
            extended_bound[dim] = mid
            bound_size = self.query_result_size(extended_bound, approximate=False)
            if bound_size < data_threshold:
                L = mid
            elif bound_size > data_threshold:
                U = mid
                if U - L < 0.00001:
                    break
            else:
                break
            if print_info:
                print("loop,L:", L, "U:", U, "mid:", mid, "extended_bound:", extended_bound, "size:", bound_size)
            loop_count += 1

        return bound_size >= data_threshold, extended_bound, bound_size

    def __is_overlap(self, boundary, query):
        '''
        the difference between this function and the public is_overlap function lies in the boundary parameter
        '''
        if len(query) != 2 * self.num_dims:
            return -1  # error

        overlap_flag = True
        inside_flag = True

        for i in range(self.num_dims):
            if query[i] >= boundary[self.num_dims + i] or query[self.num_dims + i] <= boundary[i]:
                overlap_flag = False
                inside_flag = False
                return 0
            elif query[i] < boundary[i] or query[self.num_dims + i] > boundary[self.num_dims + i]:
                inside_flag = False

        if inside_flag:
            return 2
        elif overlap_flag:
            return 1
        else:
            return 0


    def __max_bound(self, queryset):
        '''
        bound the queries by their maximum bounding rectangle !NOTE it is for a collection of queries!!!
        then constraint the MBR by the node's boundary!
        
        the return bound is in the same form as boundary
        '''
        if len(queryset) == 0:
            return None
        # if len(queryset) == 1:
        #    pass, I don't think there will be shape issue here

        max_bound_L = np.amin(np.array(queryset)[:, 0:self.num_dims], axis=0).tolist()
        # bound the lower side with the boundary's lower side
        max_bound_L = np.amax(np.array([max_bound_L, self.boundary[0:self.num_dims]]), axis=0).tolist()

        max_bound_U = np.amax(np.array(queryset)[:, self.num_dims:], axis=0).tolist()
        # bound the upper side with the boundary's upper side
        max_bound_U = np.amin(np.array([max_bound_U, self.boundary[self.num_dims:]]), axis=0).tolist()

        max_bound = max_bound_L + max_bound_U  # concat
        return max_bound
    
    def max_bound_for_query(self,q):
        query=q.copy()
        return self.__max_bound_single(query)

    def __max_bound_single(self, query, parent_boundary=None):
        '''
        bound anything in the shape of query by the current partition boundary
        '''
        if parent_boundary is None:
            for i in range(self.num_dims):
                query[i] = max(query[i], self.boundary[i])
                query[self.num_dims + i] = min(query[self.num_dims + i], self.boundary[self.num_dims + i])
            return query
        else:
            for i in range(self.num_dims):
                query[i] = max(query[i], parent_boundary[i])
                query[self.num_dims + i] = min(query[self.num_dims + i], parent_boundary[self.num_dims + i])
            return query