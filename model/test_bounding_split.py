from partition_algorithm import PartitionAlgorithm
import logging
from colorlog import ColoredFormatter

handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = ColoredFormatter(
    "%(log_color)s%(levelname)-8s:%(name)s:%(message)s",
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold_red",
    },
)
handler.setFormatter(formatter)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.handlers = []
logger.addHandler(handler)


def test_bounding_split_effect():
    """
    Test the effect of bounding split when enabled and disabled
    """
    benchmark = "tpch"
    benchmark_dict = {
        "tpch": [
            "lineitem",
            "orders",
            "customer",
            # "nation",
            # "region",
            # "part",
            # "supplier",
            # "partsupp",
        ],
        "imdb": ["title", "movie_companies", "cast_info", "name"],
        "tpcds": [
            "store", 
            "item",
            "household_demographics", 
            "customer", 
        ],
    }

    trees = {}
    pa = PartitionAlgorithm(benchmark=benchmark)

    pa.load_join_query(join_indeuced="PAW")
    cost_dict = dict()

    table_list = set()
    for join_query in pa.join_queries:
        if join_query["join_relations"]:
            for join_op in join_query["join_relations"]:
                for join_table, join_col in join_op.items():
                    table_list.add(join_table)

    for tablename in benchmark_dict[benchmark]:
        pa.table_name = tablename
        pa.load_data()
        pa.load_query(join_indeuced="PAW")

        for if_bounding_split in [True, False]:
            pa.InitializeWithJT(
                    enable_bounding_split=if_bounding_split, enable_median_extend=False
                )
            bounding_flag = 1 if if_bounding_split else 0

            cost_dict.setdefault(bounding_flag, {})
            trees.setdefault(bounding_flag, {})
            trees[bounding_flag][tablename] = pa.partition_tree
            trees[bounding_flag][tablename].name = "PAC-Tree"
            tree_depth = pa.evaluate_tree_depth(pa.partition_tree.pt_root, 0)
            tot_cost = pa.evaluate_single_table_access_cost()
            cost_dict[bounding_flag][tablename] = tot_cost
            logging.info(
                f"enable bounding: {if_bounding_split}, {cost_dict[bounding_flag][tablename]}, max_depth:{tree_depth}"
            )

    for bounding_flag in cost_dict.keys():
        avg_cost = sum(cost_dict[bounding_flag].values()) / len(cost_dict[bounding_flag])
        logger.info(
            f"Average cost for bounding split {'enabled' if bounding_flag == 1 else 'disabled'}: {avg_cost}"
        )
if __name__ == "__main__":
    test_bounding_split_effect()

"""
bounding split test result:
_______imdb_______________
___table __enable __disable
title: 0.022  0.02495
movie_companies:  0.04585  0.069358
cast_info: 0.0083 0.0083
name: 0.0414 0.041


________tpch______________
___table __enable __disable
lineitem:0.313  0.313
orders： 0.0527 0.0687
customer：0.070 0.07009


_______tpcds_______________
store: 0.17105 0.17105
item: 0.1307 0.1307
household_demographics: 0.4026 0.4026
customer: 0.314 0.3148

"""