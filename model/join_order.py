import copy
class Node:
    def __init__(self,tab_name):
        self.table=tab_name
        self.column_w={}
        self.adj={}
        self.adj_col={}
    def add_column(self,col_name,result_size):
        self.column_w[col_name]=result_size

class TraditionalJoinOrder:
    def __init__(self,join_raltions,metadata):
        paths=[]
        join_raltions_copy=copy.deepcopy(join_raltions)   
        used_nodes=[]
        for op_id,join_op in enumerate(join_raltions_copy):
            left_table,right_table=list(join_op.keys())
            left_col,right_col=metadata[left_table]['used_cols'][join_op[left_table]],metadata[right_table]['used_cols'][join_op[right_table]]
            if op_id==0:
                node1=Node(left_table)
                node2=Node(right_table)
                node1.adj_col[node2]=left_col
                node2.adj_col[node1]=right_col
                paths.append((node1,node2,1))
                used_nodes.append(node1)
                used_nodes.append(node2)
                continue
            flag=True
            left_new=False
            if left_table not in [node.table for node in used_nodes] and right_table not in [node.table for node in used_nodes]:
                join_raltions_copy.append(join_op)
                continue    
            if left_table not in [node.table for node in used_nodes]:
                node1=Node(left_table)
                used_nodes.append(node1)
                left_new=True
            else:
                node1=[node for node in used_nodes if node.table==left_table][0]
                # for item in paths:
                #     if (item[0]==node1 or item[1]==node1) and item[2]==1:
                #         flag=False
                #         break
                for pid,item in enumerate(paths):
                    if (item[0]==node1 or item[1]==node1):
                        if item[0]==node1:   #Not necessarily hyper, but old node is at the front position of join
                            flag=False
                            break
                        if item[2]==1:  #If hyper join appeared before, new node must be shuffle
                            flag=False
                            break
                if paths[-1][0]!=node1 and paths[-1][1]!=node1:
                    flag=False
                
            if right_table not in [node.table for node in used_nodes]:
                node2=Node(right_table)
                used_nodes.append(node2)
            else:
                node2=[node for node in used_nodes if node.table==right_table][0]
                for pid,item in enumerate(paths):
                    if (item[0]==node2 or item[1]==node2):
                        if item[0]==node2:
                            flag=False
                            break
                        if item[2]==1:
                            flag=False
                            break
                if paths[-1][0]!=node2 and paths[-1][1]!=node2:
                    flag=False
            node1.adj_col[node2]=left_col
            node2.adj_col[node1]=right_col
            
            if left_new:
                node1, node2 = node2, node1
            if flag:
                paths.append((node1,node2,1))
            else:
                paths.append((node1,node2,-1))
        self.paths=paths
        self.print_mst(paths)
    def print_mst(self,mst):
        join_order_str=''
        for idx,tup in enumerate(mst):
            join_type='hyper' if tup[2]==1 else 'shuffle'
            if tup[2]==1:
                join_order_str+=f"[{tup[0].table}.{tup[0].adj_col[tup[1]]}=>{join_type}=>{tup[1].table}.{tup[1].adj_col[tup[0]]}] "
            
            if tup[2]==-1:
                join_order_str+=f"({tup[0].table}.{tup[0].adj_col[tup[1]]}=>{join_type}=>{tup[1].table}.{tup[1].adj_col[tup[0]]}) "
        
        join_order_str+='\n'
                
        print(join_order_str)

class JoinGraph:
    
    def __init__(self,scan_block_dict, hyper_nodes):
        GraphJ=[]
        
        for table_col in scan_block_dict['card'].keys():
            tab_name,_=table_col.split('.')
            if tab_name not in [node.table for node in GraphJ]:
                GraphJ.append(Node(tab_name))
        for table_col,result_size in scan_block_dict['card'].items():
            tab_name,tab_column=table_col.split('.')
            node=GraphJ[[node.table for node in GraphJ].index(tab_name)]
            node.add_column(tab_column,result_size)
            
        # adj={node:{} for node in GraphJ}
        for item in scan_block_dict['relation']:
            left_table_column,right_table_column=item[0],item[1]
            left_tab_name,left_col_name=left_table_column.split('.')
            right_tab_name,right_col_name=right_table_column.split('.')
            left_node=GraphJ[[node.table for node in GraphJ].index(left_tab_name)]
            right_node=GraphJ[[node.table for node in GraphJ].index(right_tab_name)]
            # left_node.adj[right_node]=[left_col_name,right_col_name,item[2]]
            # right_node.adj[left_node]=[right_col_name,left_col_name,item[2]]
            
            left_node.adj[right_node]=self.cpt_edge_weight(left_node,left_col_name,right_node,right_col_name,item[2],hyper_nodes)
            left_node.adj_col[right_node]=left_col_name
            right_node.adj[left_node]=self.cpt_edge_weight(right_node,right_col_name,left_node,left_col_name,item[2],hyper_nodes)
            right_node.adj_col[left_node]=right_col_name
            
        self.GraphJ=GraphJ
    
    def generate_MST(self):
        
        start_edges=[]
        max_edge,max_hyper_weight=(),-1
        for node in self.GraphJ:
            if self.return_degree(node)==1: 
                start_edges.append((node,list(node.adj.keys())[0]))
            for joined_node,wrap in node.adj.items():
                if wrap['csy_w']-wrap['chy_w']>max_hyper_weight:
                    max_hyper_weight=wrap['csy_w']-wrap['chy_w']
                    max_edge=(node,joined_node)
        print(max_hyper_weight)

        if max_edge not in start_edges:
            start_edges.append(max_edge)
        # self.print_graph(start_edge)
            
        min_MST,min_MST_weight=None,float('inf')

        for start_edge in start_edges:
            paths=[(start_edge[0],start_edge[1],1)]
            total_weight=start_edge[0].adj[start_edge[1]]['chy_w']
            used_nodes=[start_edge[0],start_edge[1]]
            candidated_explore_nodes=[(start_edge[0],1),(start_edge[1],1)]

            while len(paths)+1<len(self.GraphJ):
                min_edge,min_edge_weight=None,float('inf')
                for cur_node,cur_flag in candidated_explore_nodes:
                    for joined_node,wrap in cur_node.adj.items():
                        if joined_node not in used_nodes:
                            new_flag=-cur_flag
                            if cur_flag==1:
                                cur_weight=wrap['csy_w']
                            else:
                                if paths[-1][1]==cur_node: #Ensure the front node of this hyper join is the end node of the last join
                                    cur_weight=wrap['chy_w']
                                else:
                                    cur_weight=wrap['csy_w']
                                    new_flag=cur_flag
                            if cur_weight<min_edge_weight:
                                min_edge_weight=cur_weight
                                min_edge=(cur_node,joined_node,new_flag)
                
                paths.append(min_edge)
                used_nodes.append(min_edge[1])

                if min_edge[2]==1:
                    candidated_explore_nodes.remove((min_edge[0],-min_edge[2]))
                    candidated_explore_nodes.append((min_edge[0],min_edge[2]))
                candidated_explore_nodes.append((min_edge[1],min_edge[2]))
                                
                total_weight+=min_edge_weight
            if total_weight<min_MST_weight: 
                min_MST_weight=total_weight
                min_MST=paths

        print("The minimum spanning tree is:")
        self.print_mst(min_MST)
        return min_MST
    
    def return_degree(self,node):
        return len(node.adj.keys())
    
    
    def cpt_edge_weight(self,n_x,col_x,n_y,col_y,join_type,hyper_nodes):
        if join_type=='Hyper' and  col_x in hyper_nodes[n_x.table] and col_y in hyper_nodes[n_y.table]:
            chy=(n_x.column_w[col_x]+n_y.column_w[col_y])*1.2
            csy=(n_x.column_w[col_x]+n_y.column_w[col_y]*3)
        else:
            chy=(n_x.column_w[col_x]+n_y.column_w[col_y]*3)
            csy=(n_x.column_w[col_x]+n_y.column_w[col_y]*3)
        return {'chy_w':chy,'csy_w':csy}

    def print_graph(self,paths):
        path_str=''
        for table_col in paths:
            path_str+='->'+table_col
        print(path_str)

    def print_mst(self,mst):
        join_order_str=''
        for idx,tup in enumerate(mst):
            join_type='hyper' if tup[2]==1 else 'shuffle'
            if tup[2]==1:
                join_order_str+=f"[{tup[0].table}.{tup[0].adj_col[tup[1]]}=>{join_type}=>{tup[1].table}.{tup[1].adj_col[tup[0]]}] "
            
            if tup[2]==-1:
                join_order_str+=f"({tup[0].table}.{tup[0].adj_col[tup[1]]}=>{join_type}=>{tup[1].table}.{tup[1].adj_col[tup[0]]}) "
        
        join_order_str+='\n'
                
        print(join_order_str)

