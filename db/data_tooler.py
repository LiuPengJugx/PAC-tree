import csv
import os
from connector import Connector
import argparse
import pickle

current_path = os.path.dirname(os.path.abspath(__file__))


def export_metadata(benchmark):

    dataset_base_dir = current_path + "/../dataset/" + benchmark
    # Initialize the database connector
    connector = Connector(benchmark)
    connector.connect()

    # Part 1: Get numeric and non-numeric columns and their ranges for each table
    sql_tables = """
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public'
    """
    tables = connector.execute_query(sql_tables)
    schema_details = {}

    for table in tables:
        table_name = table[0]
        sql_columns = f"""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = '{table_name}'
            ORDER BY ordinal_position
        """
        columns = connector.execute_query(sql_columns)
        numeric_cols = []
        text_cols = []
        col_ranges = {}
        col_widths = {}
        for column in columns:
            col_name = column[0]
            data_type = column[1]
            if data_type in (
                "integer",
                "bigint",
                "smallint",
                "decimal",
                "numeric",
                "real",
                "double precision",
                "serial",
                "bigserial",
                "date",
                "timestamp",
            ):
                numeric_cols.append(col_name)
            else:
                text_cols.append(col_name)

        # Compute column ranges
        for col in numeric_cols.copy():
            min_max_query = f"""
                SELECT MIN("{col}"), MAX("{col}")
                FROM "{table_name}"
            """
            min_max = connector.execute_query(min_max_query)
            if min_max:
                if min_max[0][0] is None or min_max[0][1] is None:
                    numeric_cols.remove(col)
                    text_cols.append(col)
                else:
                    col_ranges[col] = {"min": min_max[0][0], "max": min_max[0][1]}

        # Compute column widths
        for col in numeric_cols + text_cols:
            width_query = f"""
                SELECT MAX(LENGTH(CAST("{col}" AS TEXT))) 
                FROM "{table_name}"
            """
            df_width_info = connector.execute_query(width_query)
            if df_width_info and df_width_info[0][0] is not None:
                col_widths[col] = df_width_info[0][0]
            else:
                col_widths[col] = 0  # Default width if no data

        schema_details[table_name] = {
            "numeric_columns": numeric_cols,
            "text_columns": text_cols,
            "ranges": col_ranges,
            "width": col_widths,  # Add width information
            "read_line": sum([col_widths[col] for col in numeric_cols]),
            "rows": connector.execute_query(f"SELECT COUNT(*) FROM {table_name};")[0][
                0
            ],
        }

    # Output column info
    print("Table Column Information:")
    for table, info in schema_details.items():
        print(f"Table: {table}")
        print(f"  Numeric Columns:")
        for col in info["numeric_columns"]:
            rng = info["ranges"][col]
            width = info["width"][col]
            print(f"    {col}: min={rng['min']}, max={rng['max']}, width={width}")
        print(f"  Text Columns:")
        for col in info["text_columns"]:
            width = info["width"][col]
            print(f"    {col}: width={width}")
        print()

    with open(f"{dataset_base_dir}/metadata.pkl", "wb") as f:
        pickle.dump(schema_details, f)
    with open(f"{dataset_base_dir}/metadata.txt", "w") as f:
        for table, info in schema_details.items():
            f.write(f"Table: {table}\n")
            f.write(f"  Numeric Columns:\n")
            for col in info["numeric_columns"]:
                rng = info["ranges"][col]
                width = info["width"][col]
                f.write(
                    f"    {col}: min={rng['min']}, max={rng['max']}, width={width}\n"
                )
            f.write(f"  Text Columns:\n")
            for col in info["text_columns"]:
                width = info["width"][col]
                f.write(f"    {col}: width={width}\n")
            f.write("\n")

    connector.close_all_connection()


def export_tables_to_csv(benchmark):
    """
    Export all tables from the specified benchmark database to CSV files under dataset/{benchmark}/,
    including column headers.
    """
    connector = Connector(benchmark)
    connector.connect()

    # Get all tables in public schema
    tables_query = """
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public'
        
    """
    table_list = connector.execute_query(tables_query)

    output_dir = os.path.join(os.path.dirname(__file__), "../dataset", benchmark)
    os.makedirs(output_dir, exist_ok=True)

    for (table_name,) in table_list:
        select_query = f'SELECT * FROM "{table_name}"'
        conn = connector.get_connection()
        with conn.cursor() as cur:
            cur.execute(select_query)
            rows = cur.fetchall()
            column_names = [desc[0] for desc in cur.description]

        connector.close_connection(conn)
        csv_path = os.path.join(output_dir, f"{table_name}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(column_names)
            writer.writerows(rows)

    connector.close_all_connection()


def clean_csv(benchmark):
    """
    Clean null values in numeric columns of all CSV files under the specified benchmark.
    - Deletion criteria: Any row containing NULL or empty values in numeric columns will be removed.
    """
    import pandas as pd
    import numpy as np
    import pickle

    # Get dataset directory
    dataset_dir = os.path.join(current_path, "../dataset", benchmark)
    # Load metadata to identify numeric columns
    metadata_path = os.path.join(dataset_dir, "metadata.pkl")
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file does not exist at {metadata_path}")
        print("Please run --export_metadata first to generate metadata")
        return

    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    # Get all CSV files in directory
    csv_files = [f for f in os.listdir(dataset_dir) if f.endswith(".csv")]
    print(f"Starting to clean CSV files for {benchmark} benchmark...")
    # NA values dictionary
    na_values = ["", "NULL", "null", "NA", "N/A", "None", "none"]
    for csv_file in csv_files:
        if csv_file == "catalog_sales.csv":
            continue

        # Process all CSV files
        file_path = os.path.join(dataset_dir, csv_file)
        table_name = csv_file[:-4]  # Remove .csv extension
        # Check if table exists in metadata
        if table_name not in metadata:
            print(f"Skipping: {csv_file} (not found in metadata)")
            continue
        print(f"Processing: {csv_file}")
        try:
            # Read file with pandas, handle various null values
            df = pd.read_csv(
                file_path, na_values=na_values, keep_default_na=True, low_memory=False
            )
            # Get numeric columns for this table
            numeric_columns = metadata[table_name]["numeric_columns"]
            # Only consider numeric columns that actually exist in the CSV
            numeric_columns = [col for col in numeric_columns if col in df.columns]
            if not numeric_columns:
                print(
                    f"  - Warning: No numeric columns found in {csv_file}, skipping cleanup"
                )
                continue
            total_rows = len(df)
            print(f"  - Total rows: {total_rows}")
            print(f"  - Numeric columns: {', '.join(numeric_columns)}")
            # Count null values in numeric columns
            null_counts = df[numeric_columns].isnull().sum()
            print("  - Null value counts in numeric columns:")
            for col, count in null_counts.items():
                if count > 0:
                    print(
                        f"    * {col}: {count} rows with null values ({count/total_rows:.2%})"
                    )
            # Total rows with null values in numeric columns
            rows_with_numeric_nulls = df[numeric_columns].isnull().any(axis=1).sum()
            print(
                f"  - Rows with null values in numeric columns: {rows_with_numeric_nulls} ({rows_with_numeric_nulls/total_rows:.2%})"
            )
            if rows_with_numeric_nulls == 0:
                print("  - No rows with null values in numeric columns, file unchanged")
                continue
            # Remove rows with null values in any numeric column
            df_cleaned = df.dropna(subset=numeric_columns)
            # Save cleaned file without index
            df_cleaned.to_csv(file_path, index=False)
            # Report cleaning results
            removed_rows = len(df) - len(df_cleaned)
            print(f"  - Cleaned {removed_rows} rows ({removed_rows/total_rows:.2%})")

        except Exception as e:
            print(f"  - Error processing {csv_file}: {str(e)}")

    print("CSV file cleaning completed.")


# Input parameter benchmark
parser = argparse.ArgumentParser()
parser.add_argument("--benchmark", help="benchmark name", type=str, default="tpcds")
# Execute with --export_metadata to export metadata
parser.add_argument("--export_metadata", help="export metadata", action="store_true")
# Execute with --export_tables to export table data
parser.add_argument("--export_csv", help="export tables", action="store_true")
parser.add_argument("--clean_csv", help="clean CSV files", action="store_true")

args = parser.parse_args()
args.clean_csv = True

if args.export_metadata:
    export_metadata(args.benchmark)
if args.export_csv:
    export_tables_to_csv(args.benchmark)
if args.clean_csv:
    clean_csv(args.benchmark)
