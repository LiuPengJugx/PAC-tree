from model.partition_algorithm import PartitionAlgorithm
from model.partition_tree import PartitionTree
import os

base_dir= os.path.dirname(os.path.abspath(__file__))
benchmark='tpch'
# algorithm='ADP'
algorithm='JT'
      
def construct_join_trees():
    pa=PartitionAlgorithm(benchmark=benchmark)
    pa.load_join_query()
    for tablename in ['lineitem','orders','customer','nation','region','part','supplier','partsupp']:
        pa.table_name=tablename
        pa.load_data()
        pa.load_query()
        if algorithm=='ADP':
            pa.InitializeWithADP()
        else:
            pa.InitializeWithJT()
        # pa.InitializeWithQDT()
        # pa.load_tree('QDT')

def evaluate_join_trees():

    def load_tree(model,benchmark,tablename,join=False):
        path=f'{base_dir}/layouts/{benchmark}/{tablename}-{model}.pkl'
        pt=PartitionTree()
        if join:
            pt.join_attr=''
        pt.load_tree(path)
        return pt  
    pa=PartitionAlgorithm()
    pa.load_join_query()
    trees={}
    for tablename in ['lineitem','orders','customer','nation','region','part','supplier','partsupp']:
        if algorithm=='ADP':
            trees[tablename]=load_tree('ADP',benchmark,tablename,join=True)
            trees[tablename].name='AdaptDB'
        else:
            trees[tablename]=load_tree('JT',benchmark,tablename,join=True)
            trees[tablename].name='JoinTree'
    pa.evaluate_multiple_table_access_cost(trees)

# construct_join_trees()
evaluate_join_trees()
print('end')
