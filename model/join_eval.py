import random
import time
import numpy as np
import pandas as pd
from db.conf import table_suffix
"""
A tool class that provides QDG algorithm and computes hyper / shuffle join cost.
"""
class JoinEvaluator:
    def __init__(self,join_queryset, joined_cols,joined_trees,joined_tables,block_size,metadata,benchmark='tpch'):
        self.A_join_queries=[join_queryset[0]]
        self.B_join_queries=[join_queryset[1]]
        self.join_tables=joined_tables
        self.pa_A=joined_trees[0]
        self.pa_B=joined_trees[1]
        self.join_attr=joined_cols
        self.metadata=metadata
        self.benchmark=benchmark
        self.dim_nums1,self.dim_nums2=len(join_queryset[0])//2,len(join_queryset[1])//2
        self.worker_mermory=10 #每个block大小默认为2*block_size=20000 =>2*128*10=2560=>2.5G 即处理查询时，分配给每个worker 2.5G的内存，因为需要多个worker做map操作，最后放到reducer中进行join。
        # 其实worker的内存分配，通常和parallelism有关。 根据parallelism的大小，经验性地设置worker的内存大小。
        # 做实验验证从 500M到 30G的worker内存大小，对于join操作的性能影响。
        
        # 对于 100GB 的表数据，如果假设数据均匀分布在 10 个 workers 上，每个 worker 需要处理大约 10GB 的数据。考虑到 join 操作的内存开销和其他任务的内存需求，每个 worker 的内存通常会配置为数据大小的 2-3 倍，以确保有足够的内存进行计算和缓存。



    def generate_join_queries(self,a_training_set_for_join,b_training_set_for_join,join_amount=20):
        join_attr=self.join_attr
        def __overlap(q1, q2, dim1, dim2):
            if q1[dim] <= q2[dim] <= q1[dim + self.dim_nums1] or q2[dim] <= q1[dim] <= q2[dim + self.dim_nums2]:
                return True
            return False
        # a_training_set=self.a_training_set
        # b_training_set=self.b_training_set
        a_training_set=a_training_set_for_join
        b_training_set=b_training_set_for_join

        # pick join query which will be measure
        b_join_index = []  #b_join_index对应着每个
        for _ in range(join_amount):
            b_join_index.append(
                list(set([random.randint(0, len(b_training_set) - 1) for _ in range(random.randint(1, 10))])))
        # remove block id with overlap join attribute range
        b_join_queries = []
        for ids in b_join_index:
            item = []
            for idx in ids:
                flag = True
                for em in item:
                    if __overlap(b_training_set[idx], em, join_attr):
                        flag = False
                        break
                if flag: item.append(b_training_set[idx])
            b_join_queries.append(item)
        a_join_queries = {}
        
        #为每条b查询，选择所有与之overlap的a查询。
        # a_join_queries是一个字典，key是b查询的编号，value是与之overlap的a查询的集合
        for bid, item in enumerate(b_join_queries):
            for qb in item:
                a_join_queries[bid] = []
                for qa in a_training_set:
                    if __overlap(qa, qb, join_attr):
                        # remove overlap range queries
                        flag = True
                        for qa2 in a_join_queries[bid]:
                            # if __overlap(qa2,qa,join_attr):
                            if qa2 == qa:
                                flag = False
                                break
                        if flag: a_join_queries[bid].append(qa)
        # for key in a_join_queries.keys():
        #     print(f"{key} : {len(a_join_queries[key])}")
        return a_join_queries,b_join_queries


    # # 粗略估计join操作的成本
    # def rough_join_cost(self,group_type):
    #     hyper_read_cost, hyper_read_bytes, temp_joined_df, group_time= self.compute_total_shuffle_hyper_cost(group_type)
    #     # if shuffle:
    #     #     return shuffle_read_cost
    #     return hyper_read_cost,hyper_read_bytes,temp_joined_df,group_time
    
    def pandas_hash_join(self,dataset_A, dataset_B, key_A, key_B):
        A_used_columns=self.pa_A.used_columns
        B_used_columns=self.pa_B.used_columns
        if self.benchmark=='imdb':
            A_used_columns=[table_suffix[self.benchmark][self.join_tables[0]]+'_'+col for col in A_used_columns]
            B_used_columns=[table_suffix[self.benchmark][self.join_tables[1]]+'_'+col for col in B_used_columns]
        # 将 NumPy 数组转换为 DataFrame，并指定列名
        df_A = pd.DataFrame(dataset_A, columns=A_used_columns)
        df_B = pd.DataFrame(dataset_B, columns=B_used_columns)

        # 取出要用来 join 的原始列名
        left_col = A_used_columns[key_A]
        right_col = B_used_columns[key_B]

        if df_A.shape[0]>df_B.shape[0]:
            df_A,df_B=df_B,df_A
            left_col,right_col=right_col,left_col

        df_B = df_B.drop_duplicates(subset=[right_col], keep='first')
        # 指定 left_on 和 right_on 进行哈希合并
        merged_df = df_A.merge(df_B, how='inner', left_on=left_col, right_on=right_col)

        # 返回合并后的 NumPy 数组
        return merged_df

    def compute_total_shuffle_hyper_cost(self,group_type,is_real_hyper):
        start_time = time.time()
        pa_A=self.pa_A
        pa_B=self.pa_B
        key_A,key_B=self.join_attr[0],self.join_attr[1]
        blocks_A_ids = {}
        blocks_B_ids = {}
        for qid, query in enumerate(self.A_join_queries):
            blocks_A_ids[qid]=list(set(pa_A.query_single(query)))
        
        for qid, query in enumerate(self.B_join_queries):
            blocks_B_ids[qid]=list(set(pa_B.query_single(query)))

        print(f"Table {self.join_tables[0]} has {len(blocks_A_ids[0])} Blocks, Table {self.join_tables[1]} has {len(blocks_B_ids[0])} Blocks.")
        print(f"Read Cost: {sum([pa_A.nid_node_dict[a_id].node_size for a_id in blocks_A_ids[0]])} -> {sum([pa_B.nid_node_dict[b_id].node_size for b_id in blocks_B_ids[0]])}")
        static_A_blocks=blocks_A_ids[0].copy()
        static_B_blocks=blocks_B_ids[0].copy()

        # compute hyper join cost （Use Group 4）
        def is_overlay(aid, bid):
            bucket_a = pa_A.nid_node_dict[aid].boundary
            bucket_b = pa_B.nid_node_dict[bid].boundary
            return __overlap(bucket_a, bucket_b, key_A,key_B)
        def __overlap(q1, q2, dim1, dim2):
            if q1[dim1] <= q2[dim2] <= q1[dim1 + self.dim_nums1] or q2[dim2] <= q1[dim1] <= q2[dim2 + self.dim_nums2]:
                return True
            return False
        
        final_resized_splits = []
        overlap_chunks_for_queries = []
        
        build_time = 0
        for qid in range(len(blocks_A_ids)):
            A_join_block_ids=blocks_A_ids[qid]
            B_join_block_ids=blocks_B_ids[qid]
            
            # group algorithm
            # step1: generate overlap_chunks
            overlap_chunks = {}
            for A_bid in A_join_block_ids:
                if A_bid not in overlap_chunks.keys(): 
                    overlap_chunks[A_bid] = []
                for B_bid in B_join_block_ids:
                    if is_overlay(A_bid, B_bid): 
                        overlap_chunks[A_bid].append(B_bid)
                    
            overlap_chunks_for_queries.append(overlap_chunks)
            # step2: group
            time0 = time.time()
            if group_type==0:
                resizedSplits = self.group(overlap_chunks, A_join_block_ids, min_partition_size=self.worker_mermory)
            elif group_type==1:
                resizedSplits = self.group1(overlap_chunks, A_join_block_ids, min_partition_size=self.worker_mermory,max_partition_size=1.2*self.worker_mermory)
            build_time += time.time() - time0
            final_resized_splits.append(resizedSplits)
        group_time = time.time() - start_time
        total_shuffle_hyper_read_cost = 0
        a_hyper_cost,b_hyper_cost=0,0
        a_shuffle_cost,b_shuffle_cost=0,0
        shuffle_weight = 3
        
        for q_no, resizedSplits in enumerate(final_resized_splits):
            total_A_ids = []
            total_B_ids = []
            overlap_chunks = overlap_chunks_for_queries[q_no]
            group_a_cost,group_b_cost=0,0
            for group in resizedSplits:
                B_ids = []
                for a_id in group:
                    group_a_cost+=pa_A.nid_node_dict[a_id].node_size
                    B_ids += overlap_chunks[a_id]
                    
                total_A_ids+=group
                B_ids = list(set(B_ids))
                total_B_ids += B_ids
                for b_id in B_ids:
                    group_b_cost+=pa_B.nid_node_dict[b_id].node_size

            a_hyper_cost+=group_a_cost
            b_hyper_cost+=group_b_cost
            
            total_B_ids = list(set(total_B_ids))
            for a_id in total_A_ids:
                a_shuffle_cost += shuffle_weight * pa_A.nid_node_dict[a_id].node_size
            for b_id in total_B_ids:
                b_shuffle_cost += shuffle_weight * pa_B.nid_node_dict[b_id].node_size

        # 获取连接结果
        dataset_A,dataset_B=[],[]
        for a_id in static_A_blocks:
            dataset_A.extend(pa_A.nid_node_dict[a_id].dataset)
        for b_id in static_B_blocks:
            dataset_B.extend(pa_B.nid_node_dict[b_id].dataset)
        joined_df=self.pandas_hash_join(np.array(dataset_A),np.array(dataset_B),key_A,key_B)
        if is_real_hyper:
            total_shuffle_hyper_read_cost=a_hyper_cost+b_hyper_cost
            total_hyper_shuffle_read_bytes=a_hyper_cost*self.metadata[self.join_tables[0]]['read_line']+b_hyper_cost*self.metadata[self.join_tables[1]]['read_line']
        else:
            total_shuffle_hyper_read_cost=a_shuffle_cost+b_shuffle_cost
            total_hyper_shuffle_read_bytes=a_shuffle_cost*self.metadata[self.join_tables[0]]['read_line']+b_shuffle_cost*self.metadata[self.join_tables[1]]['read_line']
        print(f"Hyper Join: {a_hyper_cost} -> {b_hyper_cost}")
        print(f"Real Join Type: {is_real_hyper}, Final Cost: {total_shuffle_hyper_read_cost}")

        return total_shuffle_hyper_read_cost,total_hyper_shuffle_read_bytes,joined_df,group_time
    
    
    

    def print_shuffle_hyper_blocks(self,a_join_queries,b_join_queries,group_type):
        pa_A=self.pa_A
        pa_B=self.pa_B
        join_attr=self.join_attr
        blocks_a_ids = []
        blocks_b_ids = []
        a_join_info = []
        b_join_info = []
        # how to get join attr range base on block id.
        for key, queries in enumerate(b_join_queries):
            map_content = {}
            join_keys = []
            block_ids = []
            for query in queries:
                join_keys += pa_B.query_single_toplayer(query)
                block_ids += pa_B.query_single(query)
            map_content[key] = list(set(block_ids))
            blocks_b_ids.append(map_content)

            join_keys = list(set(join_keys))
            join_info = {"nums": len(join_keys), "length": []}
            for join_id in join_keys:
                node = pa_B.nid_node_dict[join_id]
                join_info["length"].append(node.boundary[join_attr + node.num_dims] - node.boundary[join_attr])
            b_join_info.append(join_info)

        for key in a_join_queries:
            map_content = {}
            join_keys = []
            block_ids = []
            for query in a_join_queries[key]:
                join_keys += pa_A.query_single_toplayer(query)
                block_ids += pa_A.query_single(query)
            map_content[key] = list(set(block_ids))
            blocks_a_ids.append(map_content)

            join_keys = list(set(join_keys))
            join_info = {"nums": len(join_keys), "length": []}
            for join_id in join_keys:
                node = pa_A.nid_node_dict[join_id]
                join_info["length"].append(node.boundary[join_attr + node.num_dims] - node.boundary[join_attr])
            a_join_info.append(join_info)

        join_infos = [a_join_info, b_join_info]
        for join_info in join_infos:
            total_nums, total_length = 0, 0
            for item in join_info:
                total_nums += item['nums']
                total_length += sum(item['length'])
            # print(f"{total_nums} , {round(total_length, 2)}")

        # compute hyper join cost （Use Group 4）
        def is_overlay(aid, bid):
            bucket_a = pa_A.nid_node_dict[aid].boundary
            bucket_b = pa_B.nid_node_dict[bid].boundary
            return __overlap(bucket_a, bucket_b, join_attr)
        def __overlap(q1, q2, dim):
            if q1[dim] <= q2[dim] <= q1[dim + self.dim_nums1] or q2[dim] <= q1[dim] <= q2[dim + self.dim_nums2]:
                return True
            return False
        final_resized_splits = []
        overlap_chunks_for_queries = []
        intersection_reward = 0
        total_hyper_cost = 0
        build_time = 0
        for idx in range(len(blocks_a_ids)):
            # if idx<=2:continue
            A_join_block_ids = []
            for key in blocks_a_ids[idx].keys():
                A_join_block_ids += blocks_a_ids[idx][key]
            B_join_block_ids = []
            for key in blocks_b_ids[idx].keys():
                B_join_block_ids += blocks_b_ids[idx][key]
            # group algorithm
            # step1: generate overlap_chunks
            overlap_chunks = {}
            for aid in A_join_block_ids:
                if aid not in overlap_chunks.keys(): overlap_chunks[aid] = []
                for bid in B_join_block_ids:
                    if is_overlay(aid, bid): overlap_chunks[aid].append(bid)
            # print(f"overlap chunks: ",overlap_chunks)
            overlap_chunks_for_queries.append(overlap_chunks)
            # step2: group
            # print(overlap_chunks)
            # print(A_join_block_ids)
            time0 = time.time()
            if group_type==3:
                resizedSplits = self.group3(overlap_chunks, A_join_block_ids, partition_size=8)
            elif group_type==1:
                resizedSplits = self.group1(overlap_chunks, A_join_block_ids, partition_size=8)

            build_time += time.time() - time0
            for group in resizedSplits:
                all_b_ids = []
                for a_id in group:
                    all_b_ids += overlap_chunks[a_id]
                    # print(overlap_chunks[a_id])
                actual_b_ids = list(set(all_b_ids))
                intersection_reward += len(all_b_ids) - len(actual_b_ids)
                total_hyper_cost += sum([pa_B.nid_node_dict[_].node_size for _ in actual_b_ids])
            final_resized_splits.append(resizedSplits)
        # print("total_hyper_cost: ", total_hyper_cost)
        # print("average build time: ", build_time / len(blocks_a_ids))
        A_ids_for_q, B_ids_for_q = [], []
        for q_no, resizedSplits in enumerate(final_resized_splits):
            cnt = 0
            total_B_ids = []
            total_A_ids = []
            overlap_chunks = overlap_chunks_for_queries[q_no]
            for group in resizedSplits:
                b_ids = []
                for a_id in group:
                    b_ids += overlap_chunks[a_id]
                    cnt += 1
                total_A_ids += group
                b_ids = list(set(b_ids))
                if b_ids:
                    total_B_ids.append(b_ids)
            A_ids_for_q.append(total_A_ids)
            B_ids_for_q.append(total_B_ids)
        return A_ids_for_q, B_ids_for_q

    def compute_join_blocks_for_main_table(self,a_join_queries,b_join_queries):
        pa_A=self.pa_A
        pa_B=self.pa_B
        join_attr=self.join_attr
        blocks_a_ids = []
        blocks_b_ids = []
        a_join_info = []
        b_join_info = []
        # how to get join attr range base on block id.
        for key, queries in enumerate(b_join_queries):
            map_content = {}
            join_keys = []
            block_ids = []
            for query in queries:
                join_keys += pa_B.query_single_toplayer(query)
                block_ids += pa_B.query_single(query)
            map_content[key] = list(set(block_ids))
            blocks_b_ids.append(map_content)

            join_keys = list(set(join_keys))
            join_info = {"nums": len(join_keys), "length": []}
            for join_id in join_keys:
                node = pa_B.nid_node_dict[join_id]
                join_info["length"].append(node.boundary[join_attr + node.num_dims] - node.boundary[join_attr])
            b_join_info.append(join_info)

        for key in a_join_queries:
            map_content = {}
            join_keys = []
            block_ids = []
            for query in a_join_queries[key]:
                join_keys += pa_A.query_single_toplayer(query)
                block_ids += pa_A.query_single(query)
            map_content[key] = list(set(block_ids))
            blocks_a_ids.append(map_content)

            join_keys = list(set(join_keys))
            join_info = {"nums": len(join_keys), "length": []}
            for join_id in join_keys:
                node = pa_A.nid_node_dict[join_id]
                join_info["length"].append(node.boundary[join_attr + node.num_dims] - node.boundary[join_attr])
            a_join_info.append(join_info)

        # print(sum([len(group_ids[key]) for key, group_ids in enumerate(blocks_a_ids)]))
        # print(sum([len(group_ids[key]) for key, group_ids in enumerate(blocks_b_ids)]))

        join_infos = [a_join_info, b_join_info]
        for join_info in join_infos:
            total_nums, total_length = 0, 0
            for item in join_info:
                total_nums += item['nums']
                total_length += sum(item['length'])
        a_hyper_blocks_size=0
        b_hyper_blocks_size=0
        for idx in range(len(blocks_a_ids)):
            # if idx<=2:continue
            for key in blocks_a_ids[idx].keys():
                a_hyper_blocks_size+=sum([pa_A.nid_node_dict[ida].node_size for ida in blocks_a_ids[idx][key]])
            for key in blocks_b_ids[idx].keys():
                b_hyper_blocks_size+=sum([pa_B.nid_node_dict[idb].node_size for idb in blocks_b_ids[idx][key]])
        return a_hyper_blocks_size,0

    # (AdaptDB grouping)
    def group(self,overlap_chunks,join_a_block_ids, min_partition_size):
        def get_intersection_size_count(setValues, listValues):
            size = 0
            for lv in listValues:
                if lv in setValues: size += 1
            return size
        
        def get_group_size(a_ids):
            chunks = []
            for a_id in a_ids:
                chunks += overlap_chunks[a_id]
            chunks = list(set(chunks))
            return sum([self.pa_B.nid_node_dict[b_id].node_size for b_id in chunks])+sum([self.pa_A.nid_node_dict[a_id].node_size for a_id in a_ids])
        
        def __get_group_size(a_ids):
            # return sum([self.pa_A.nid_node_dict[a_id].node_size for a_id in a_ids])
            return len(a_ids)
        
        resizedSplits = []
        rest_count = len(join_a_block_ids)
        while rest_count > 0:
            cur_splits = []
            chunks = []
            available_size=min_partition_size
            # max block size limit for every split.
            while rest_count > 0:
                maxIntersection = -1
                best_offset = -1
                for offset, bid in enumerate(join_a_block_ids):
                    cur_intersection = get_intersection_size_count(chunks, overlap_chunks[bid])
                    if cur_intersection > maxIntersection:
                        maxIntersection = cur_intersection
                        best_offset = offset
                bucket_id = join_a_block_ids[best_offset]
                # if _get_group_size(cur_splits+[bucket_id]) > available_size:
                #     if len(cur_splits) == 0:
                #         cur_splits.append(bucket_id)
                #         chunks += overlap_chunks[bucket_id]
                #         chunks = list(set(chunks))
                #         join_a_block_ids.remove(bucket_id)
                #         rest_count -= 1
                #         break
                #     break
                cur_splits.append(bucket_id)
                chunks += overlap_chunks[bucket_id]
                chunks = list(set(chunks))
                join_a_block_ids.remove(bucket_id)
                rest_count -= 1
                if __get_group_size(cur_splits) == available_size:
                    break
            resizedSplits.append(cur_splits)
        return resizedSplits

    # Our QDG grouping algorithm
    def group1(self, overlap_chunks, join_a_block_ids, min_partition_size, max_partition_size):
        def list_solved_list(l1, l2):
            for item1 in l1:
                if item1 in l2:
                    return True
            return False
        
        def get_group_size(a_ids):
            chunks = []
            for a_id in a_ids:
                chunks += overlap_chunks[a_id]
            chunks = list(set(chunks))
            return sum([self.pa_B.nid_node_dict[b_id].node_size for b_id in chunks])
        
        def __get_group_size(a_ids):
            # return sum([self.pa_A.nid_node_dict[a_id].node_size for a_id in a_ids])
            return len(a_ids)

        def get_intersection_size_count(setValues, listValues):
            size = 0
            for lv in listValues:
                if lv in setValues: size += 1
            return size

        resizedSplits = []
        rest_count = len(join_a_block_ids)
        affinity_tab = []
        pre_save_ids = []
        computed_ids_dict = {}
        for bid in join_a_block_ids: computed_ids_dict[bid] = {}
        a_block_len = len(join_a_block_ids)
        for no1 in range(a_block_len):
            bid1 = join_a_block_ids[no1]
            max_intersection = -1
            max_bid = []
            for exist_bid in computed_ids_dict[bid1].keys():
                cur_intersection = computed_ids_dict[bid1][exist_bid]
                if cur_intersection > max_intersection:
                    max_intersection = cur_intersection
                    max_bid = [exist_bid]
            for no2 in range(no1 + 1, a_block_len):
                bid2 = join_a_block_ids[no2]
                cur_intersection = get_intersection_size_count(overlap_chunks[bid1], overlap_chunks[bid2])
                computed_ids_dict[bid1][bid2] = cur_intersection
                computed_ids_dict[bid2][bid1] = cur_intersection
                if cur_intersection > max_intersection:
                    max_intersection = cur_intersection
                    max_bid = [bid2]
            if max_intersection == 0:
                pre_save_ids.append(bid1)
            else:
                affinity_tab.append({'item': [[bid1], max_bid], 'val': max_intersection, 'chunk': overlap_chunks[bid1]})
        cur_index = 0
        # pre-save these ids which doesn't have any overlap blocks
        last_index=0
        while last_index < len(pre_save_ids):
            # if _get_group_size(pre_save_ids[last_index:cur_index+1]) > min_partition_size:
            #     merge_ids = pre_save_ids[last_index:cur_index]
            #     resizedSplits.append(merge_ids)
            #     rest_count -= len(merge_ids)
            #     last_index=cur_index
            if __get_group_size(pre_save_ids[last_index:cur_index + 1]) == min_partition_size:
                merge_ids = pre_save_ids[last_index:cur_index + 1]
                resizedSplits.append(merge_ids)
                rest_count -= len(merge_ids)
                last_index = cur_index + 1
            
            elif cur_index >= len(pre_save_ids)-1:
                merge_ids = pre_save_ids[last_index:]
                resizedSplits.append(merge_ids)
                rest_count -= len(merge_ids)
                break
            else:
                cur_index += 1
        while rest_count > 0:
            affinity_tab.sort(key=lambda item: (item['val'], len(item['item'][0])), reverse=True)
            sel_tab = affinity_tab.pop(0)
            merge_ids = sel_tab['item'][0] + sel_tab['item'][1]
            merge_ids_length = len(merge_ids)
            is_completed = False
            # if _get_group_size(merge_ids) >= min_partition_size or len(affinity_tab) == 0 or sel_tab['val'] == -1: #这个一般不出错，因为最初两个块很难超过尺寸限制，后续合并也进行了限制。
            #     is_completed = True
            #     resizedSplits.append(merge_ids)
            #     rest_count -= merge_ids_length
                
            if __get_group_size==min_partition_size or len(affinity_tab)==0 or sel_tab['val']==-1:
                is_completed = True
                resizedSplits.append(merge_ids)
                rest_count -= merge_ids_length
            else:
                # add key=chunk
                new_overlap_chunks = sel_tab['chunk']
                for bid in sel_tab['item'][1]:
                    new_overlap_chunks += overlap_chunks[bid]
                new_tab = {'item': [merge_ids, []], 'val': -1, 'chunk': list(set(new_overlap_chunks))}

            # update affinity_tab
            for tab in reversed(affinity_tab):
                # delete tab
                if list_solved_list(tab['item'][0], sel_tab['item'][1]):
                    affinity_tab.remove(tab)
                    continue
                # update tab
                if list_solved_list(tab['item'][1], merge_ids):
                    # if is_completed or _get_group_size(tab['item'][0]) + _get_group_size(merge_ids) > max_partition_size:
                    #     tab['item'][1] = []
                    #     tab['val'] = -1
                    if is_completed or __get_group_size(tab['item'][0]) + __get_group_size(merge_ids) > min_partition_size:
                        tab['item'][1] = []
                        tab['val'] = -1
                    else:
                        tab['item'][1] = merge_ids
                        tab['val'] = get_intersection_size_count(tab['chunk'], new_tab['chunk'])
            if not is_completed: affinity_tab.append(new_tab)
            # Case: the affinity_tab only has one item.
            if len(affinity_tab) == 1:
                last_tab = affinity_tab.pop(0)
                merge_ids = last_tab['item'][0] + last_tab['item'][1]
                resizedSplits.append(merge_ids)
                rest_count -= len(merge_ids)
            # 更新merged column group的overlap_chunks和交叉收益
            for ud_item1 in affinity_tab:
                if ud_item1['val'] == -1:
                    ud1_key = ud_item1['item'][0]
                    flag1 = False
                    if len(ud1_key) == 1: flag1 = True
                    overlap_chunks1 = ud_item1['chunk']
                    max_allocate_size = min_partition_size - __get_group_size(ud1_key)
                    max_intersection = -1
                    max_target_ids = []
                    for ud_item2 in affinity_tab:
                        ud2_key = ud_item2['item'][0]
                        if ud1_key == ud2_key: continue
                        if __get_group_size(ud2_key) > max_allocate_size: continue
                        if ud_item2['item'][1] == ud1_key:
                            cur_intersection = ud_item2['val']
                        else:
                            # if flag1 and len(ud2_key)==1:continue
                            flag2 = False
                            if len(ud2_key) == 1: flag2 = True
                            if flag1 and flag2:
                                cur_intersection = computed_ids_dict[ud1_key[0]][ud2_key[0]]
                            else:
                                overlap_chunks2 = ud_item2['chunk']
                                cur_intersection = get_intersection_size_count(overlap_chunks1, overlap_chunks2)
                        if cur_intersection > max_intersection:
                            max_intersection = cur_intersection
                            max_target_ids = ud2_key
                    ud_item1['val'] = max_intersection
                    ud_item1['item'][1] = max_target_ids
        return resizedSplits



