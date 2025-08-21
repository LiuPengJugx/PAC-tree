#!/usr/bin/env python3
"""
test_pando.py - Compare the performance of Pando and MTO layouts under TPC-H workload

This script compares the following metrics:
1. Data scan ratio (scanned data volume / total data volume)
2. Number of distribution trees created by layouts
3. Number of blocks instantiated in each table
4. Block tuple utilization rate (number of tuples / max block size)
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(base_dir, ".."))

from model.partition_algorithm import PartitionAlgorithm


class PandoMTOComparator:
    def __init__(self, benchmark="tpch", scale_factor=1):
        self.benchmark = benchmark
        self.scale_factor = scale_factor
        self.queries = []
        self.tables = {}

    def build_mto_layout(self, table_name, block_size=10000):
        """Build MTO layout - using standard process of PartitionAlgorithm"""
        print(f"Building MTO layout for {table_name}...")
        pa = PartitionAlgorithm(
            block_size=block_size, benchmark=self.benchmark, table_name=table_name
        )
        pa.load_data()
        pa.load_query(join_indeuced="MTO")

        pa.used_columns = list(range(len(pa.used_columns))) 
        pa.InitializeWithQDT()
        return pa

    def build_pando_layout(self, table_name, max_depth=10, block_size=10000):
        print(f"Building Pando layout for {table_name}...")
        pa = PartitionAlgorithm(
            block_size=block_size, benchmark=self.benchmark, table_name=table_name
        )
        pa.load_data()
        pa.load_query(join_indeuced="MTO")
        pa.InitializeWithPando(max_depth=max_depth)

        return pa

    def calculate_scan_ratio(self, pa, queries, table_name):
        """计算scan ratio"""
        if not queries:
            return 0

        total_rows = len(pa.tabledata)
        total_scanned = 0

        for query in queries:
            if hasattr(pa, "partition_tree"):
                try:
                    blocks = pa.partition_tree.query_single(query)
                    scanned_rows = len(blocks) * pa.block_size
                except:
                    scanned_rows = total_rows
            else:
                try:
                    blocks = pa.query_pando(query)
                    avg_block_size = 1000  # 默认块大小
                    scanned_rows = len(blocks) * avg_block_size
                except:
                    scanned_rows = total_rows

            total_scanned += min(scanned_rows, total_rows)

        return total_scanned / (total_rows * len(queries)) if queries else 0

    def count_blocks(self, pa):
        """Count the number of blocks"""
        try:
            if hasattr(pa, "partition_tree") and pa.partition_tree:
                return len(pa.partition_tree.get_leaves())
            elif hasattr(pa, "pando_trees") and pa.pando_trees:
                total_leafs = 1
                for tree in pa.pando_trees:
                    total_leafs *= len(tree.get_leaves())
                return total_leafs
            else:
                return 1
        except:
            return 1

    def calculate_block_utilization(self, pa):
        """Calculate block utilization"""
        if hasattr(pa, "partition_tree") and pa.partition_tree:
            total_leafs = pa.partition_tree.get_leaves()
            max_tuples = (
                len(total_leafs) * pa.block_size * 2
            ) 
            total_tuples = sum([leaf.node_size for leaf in total_leafs])
            return total_tuples / max_tuples if max_tuples > 0 else 0
        elif hasattr(pa, "pando_trees") and pa.pando_trees:
            total_tuples, max_tuples = 0, 0
            total_leafs = 1
            for tree in pa.pando_trees:
                total_leafs *= len(tree.get_leaves())
            max_tuples = total_leafs * pa.block_size * 2
            total_tuples = tree.pt_root.node_size
            return total_tuples / max_tuples if max_tuples > 0 else 0
        else:
            return 0

    def run_comparison(self):
        """Run complete comparison experiment - reuse standard process of PartitionAlgorithm"""
        print("=== TPC-H Pando vs MTO Layout Comparison ===")
        tpch_tables = [
            "lineitem",
            "orders",
            "supplier",
            "customer",
            "nation",
            "region",
            "part",
            "partsupp",
        ]
        results = []
        for table_name in tpch_tables[:1]:
            print(f"\n--- Analyzing {table_name} ---")
            try:
                mto_pa = self.build_mto_layout(table_name)
                pando_pa = self.build_pando_layout(table_name)
                mto_scan_ratio = mto_pa.evaluate_single_table_access_cost()
                pando_scan_ratio = pando_pa.evaluate_pando_cost()
                mto_tree_count = 1 
                pando_tree_count = len(pando_pa.pando_trees)
                mto_blocks = self.count_blocks(mto_pa)
                pando_blocks = self.count_blocks(pando_pa)
                mto_utilization = self.calculate_block_utilization(mto_pa)
                pando_utilization = self.calculate_block_utilization(pando_pa)
                results.append(
                    {
                        "table": table_name,
                        "mto_scan_ratio": mto_scan_ratio,
                        "pando_scan_ratio": pando_scan_ratio,
                        "mto_trees": mto_tree_count,
                        "pando_trees": pando_tree_count,
                        "mto_blocks": mto_blocks,
                        "pando_blocks": pando_blocks,
                        "mto_utilization": mto_utilization,
                        "pando_utilization": pando_utilization,
                    }
                )

                print(
                    f"  Scan Ratio - MTO: {mto_scan_ratio:.4f}, Pando: {pando_scan_ratio:.4f}"
                )
                print(
                    f"  Tree Count - MTO: {mto_tree_count}, Pando: {pando_tree_count}"
                )
                print(f"  Block Count - MTO: {mto_blocks}, Pando: {pando_blocks}")
                print(
                    f"  Utilization - MTO: {mto_utilization:.4f}, Pando: {pando_utilization:.4f}"
                )

            except Exception as e:
                print(f"  Error processing {table_name}: {e}")
                raise e

        if not results:
            print("No results generated.")

        df = pd.DataFrame(results)
        self.print_summary(df)

        return df


    def print_summary(self, df):
        print("\n=== Summary Statistics ===")
        print(f"Total tables analyzed: {len(df)}")
        if len(df) > 0:
            print(f"Average scan ratio - MTO: {df['mto_scan_ratio'].mean():.4f}")
            print(f"Average scan ratio - Pando: {df['pando_scan_ratio'].mean():.4f}")
            print(f"Average tree count - MTO: {df['mto_trees'].mean():.2f}")
            print(f"Average tree count - Pando: {df['pando_trees'].mean():.2f}")
            print(f"Average block count - MTO: {df['mto_blocks'].mean():.2f}")
            print(f"Average block count - Pando: {df['pando_blocks'].mean():.2f}")
            print(f"Average utilization - MTO: {df['mto_utilization'].mean():.4f}")
            print(f"Average utilization - Pando: {df['pando_utilization'].mean():.4f}")


def main():
    comparator = PandoMTOComparator(benchmark="tpch", scale_factor=1)
    comparator.run_comparison()

    # # try:
    # #     import matplotlib.pyplot as plt

    # #     fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # #     # Scan Ratio对比
    # #     axes[0, 0].bar(
    # #         results["table"], results["mto_scan_ratio"], alpha=0.7, label="MTO"
    # #     )
    # #     axes[0, 0].bar(
    # #         results["table"], results["pando_scan_ratio"], alpha=0.7, label="Pando"
    # #     )
    # #     axes[0, 0].set_title("Scan Ratio Comparison")
    # #     axes[0, 0].set_ylabel("Scan Ratio")
    # #     axes[0, 0].legend()
    # #     axes[0, 0].tick_params(axis="x", rotation=45)

    # #     # Block Count对比
    # #     axes[0, 1].bar(results["table"], results["mto_blocks"], alpha=0.7, label="MTO")
    # #     axes[0, 1].bar(
    # #         results["table"], results["pando_blocks"], alpha=0.7, label="Pando"
    # #     )
    # #     axes[0, 1].set_title("Block Count Comparison")
    # #     axes[0, 1].set_ylabel("Block Count")
    # #     axes[0, 1].legend()
    # #     axes[0, 1].tick_params(axis="x", rotation=45)

    # #     # Utilization对比
    # #     axes[1, 0].bar(
    # #         results["table"], results["mto_utilization"], alpha=0.7, label="MTO"
    # #     )
    # #     axes[1, 0].bar(
    # #         results["table"], results["pando_utilization"], alpha=0.7, label="Pando"
    # #     )
    # #     axes[1, 0].set_title("Block Utilization Comparison")
    # #     axes[1, 0].set_ylabel("Utilization")
    # #     axes[1, 0].legend()
    # #     axes[1, 0].tick_params(axis="x", rotation=45)

    # #     # Tree Count对比
    # #     axes[1, 1].bar(results["table"], results["mto_trees"], alpha=0.7, label="MTO")
    # #     axes[1, 1].bar(
    # #         results["table"], results["pando_trees"], alpha=0.7, label="Pando"
    # #     )
    # #     axes[1, 1].set_title("Tree Count Comparison")
    # #     axes[1, 1].set_ylabel("Tree Count")
    # #     axes[1, 1].legend()
    # #     axes[1, 1].tick_params(axis="x", rotation=45)

    # #     plt.tight_layout()
    # #     plt.savefig(os.path.join(base_dir, "pando_mto_comparison.png"))
    # #     plt.close()
    # #     print("Visualization saved to pando_mto_comparison.png")

    # except ImportError:
    #     print("matplotlib not available, skipping visualization")


if __name__ == "__main__":
    main()
