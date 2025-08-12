def formatting_csv():
    import pandas as pd
    import os

    base_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = f"{base_dir}/result"

    # 1) Format query_cost.csv
    query_df = pd.read_csv(f"{result_dir}/query_cost.csv")  # Model,Query ID,Join Cost
    pivot_query = query_df.pivot(index="Query ID", columns="Model", values="Join Cost").reset_index()
    # pivot_query.columns = ["QUERY ID", "MTO", "ADP", "JT"]

    # 2) Format table_cost.csv
    table_df = pd.read_csv(f"{result_dir}/table_cost.csv")  # Model,Table,Shuffle Times,Hyper Times,Join Cost
    # Pivot for shuffle times
    pivot_shuffle = table_df.pivot_table(index="Table", columns="Model", values="Shuffle Times", aggfunc="first")
    # pivot_shuffle.columns = ["MTO", "ADP", "JT"]
    pivot_shuffle = pivot_shuffle.reset_index()

    # Pivot for hyper times
    pivot_hyper = table_df.pivot_table(index="Table", columns="Model", values="Hyper Times", aggfunc="first")
    # pivot_hyper.columns = ["MTO", "ADP", "JT"]
    pivot_hyper = pivot_hyper.reset_index()

    # Pivot for join cost
    pivot_join_cost = table_df.pivot_table(index="Table", columns="Model", values="Join Cost", aggfunc="first")
    # pivot_join_cost.columns = ["MTO", "ADP", "JT"]
    pivot_join_cost = pivot_join_cost.reset_index()

    # Merge these pivots on "Table"
    merged_df = pivot_shuffle.merge(pivot_hyper, on="Table", suffixes=(None, "_hyper"))
    merged_df = merged_df.merge(pivot_join_cost, on="Table", suffixes=(None, "_cost"))

    # 3) Write to Excel
    with pd.ExcelWriter(f"{result_dir}/formatted_results.xlsx") as writer:
        pivot_query.to_excel(writer, sheet_name="Query_Cost", index=False)
        merged_df.to_excel(writer, sheet_name="Table_Cost", index=False)

formatting_csv()