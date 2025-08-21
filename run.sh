# Generate benchmark workload
python ./db/data_tooler.py --benchmark=tpch --export_metadata --export_csv
python ./db/data_tooler.py --benchmark=imdb --export_metadata --export_csv
python ./db/data_tooler.py --benchmark=tpcds --export_metadata --export_csv

python ./db/query_tooler.py --benchmark=tpch --format --export_mto_paw_queries --export_pac_queries
python ./db/query_tooler.py --benchmark=imdb --format --export_mto_paw_queries --export_pac_queries
python ./db/query_tooler.py --benchmark=tpcds --format --export_mto_paw_queries --export_pac_queries


# Create base layout QD-TREE and candidate JoinTree
python ./model/join_key_selector.py --init=True --benchmark=tpch
python ./model/join_key_selector.py --init=True --benchmark=imdb
python ./model/join_key_selector.py --init=True --benchmark=tpcds

# Run model and output layout evaluation results
python ./model/join_key_selector.py --mode=dubug --benchmark=tpch --command=0
python ./model/join_key_selector.py --mode=dubug --benchmark=imdb --command=0
python ./model/join_key_selector.py --mode=dubug --benchmark=tpcds --command=0

# Run scaling experiments
python ./model/join_key_selector.py --mode=dubug --benchmark=tpch --command=1
python ./model/join_key_selector.py --mode=dubug --benchmark=tpch --command=2
python ./model/join_key_selector.py --mode=dubug --benchmark=tpch --command=3