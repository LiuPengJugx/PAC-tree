# benchmark workload生成
python ./db/data_tooler.py --benchmark=tpch --export_metadata --export_csv
python ./db/data_tooler.py --benchmark=imdb --export_metadata --export_csv
python ./db/data_tooler.py --benchmark=tpcds --export_metadata --export_csv

python ./db/query_tooler.py --benchmark=tpch --format --export_mto_paw_queries --export_pac_queries
python ./db/query_tooler.py --benchmark=imdb --format --export_mto_paw_queries --export_pac_queries
python ./db/query_tooler.py --benchmark=tpcds --format --export_mto_paw_queries --export_pac_queries


# 创建基础布局 QD-TREE 和 候选JoinTree
python ./model/join_key_selector.py --init=True --benchmark=tpch
python ./model/join_key_selector.py --init=True --benchmark=imdb
python ./model/join_key_selector.py --init=True --benchmark=tpcds

# 运行模型 输出布局评测结果
python ./model/join_key_selector.py --mode=dubug --benchmark=tpch --command=0
python ./model/join_key_selector.py --mode=dubug --benchmark=imdb --command=0
python ./model/join_key_selector.py --mode=dubug --benchmark=tpcds --command=0


