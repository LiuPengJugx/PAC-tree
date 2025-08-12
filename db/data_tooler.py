import csv
import os
from connector import Connector
import argparse
import pickle
current_path=os.path.dirname(os.path.abspath(__file__))

def export_metadata(benchmark):
    
    dataset_base_dir=current_path+'/../dataset/'+benchmark
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
                'integer', 'bigint', 'smallint', 'decimal', 'numeric',
                'real', 'double precision', 'serial', 'bigserial', 'date', 'timestamp'
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
                    col_ranges[col] = {'min': min_max[0][0], 'max': min_max[0][1]}
        
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
            'numeric_columns': numeric_cols,
            'text_columns': text_cols,
            'ranges': col_ranges,
            'width': col_widths,  # Add width information
            'read_line': sum([col_widths[col] for col in numeric_cols]),
            'rows': connector.execute_query(f"SELECT COUNT(*) FROM {table_name};")[0][0]
        }

    # Output column info
    print("Table Column Information:")
    for table, info in schema_details.items():
        print(f"Table: {table}")
        print(f"  Numeric Columns:")
        for col in info['numeric_columns']:
            rng = info['ranges'][col]
            width = info['width'][col]
            print(f"    {col}: min={rng['min']}, max={rng['max']}, width={width}")
        print(f"  Text Columns:")
        for col in info['text_columns']:
            width = info['width'][col]
            print(f"    {col}: width={width}")
        print()
        
    # 保存schema_details结果到.pkl文件
    with open(f'{dataset_base_dir}/metadata.pkl', 'wb') as f:
        pickle.dump(schema_details, f)
    # 同时，写一个副本，保存到txt文件中
    with open(f'{dataset_base_dir}/metadata.txt', 'w') as f:
        for table, info in schema_details.items():
            f.write(f"Table: {table}\n")
            f.write(f"  Numeric Columns:\n")
            for col in info['numeric_columns']:
                rng = info['ranges'][col]
                width = info['width'][col]
                f.write(f"    {col}: min={rng['min']}, max={rng['max']}, width={width}\n")
            f.write(f"  Text Columns:\n")
            for col in info['text_columns']:
                width = info['width'][col]
                f.write(f"    {col}: width={width}\n")
            f.write("\n")

    connector.close_all_connection()



def export_tables_to_csv(benchmark):
    """
    根据benchmark，将对应数据库中的所有表数据导出到dataset/{benchmark}/下的CSV文件，包含列名
    """
    connector = Connector(benchmark)
    connector.connect()

    # 获取所有public模式下的表
    tables_query = """
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public'
        
    """
    table_list = connector.execute_query(tables_query)

    output_dir = os.path.join(os.path.dirname(__file__), '../dataset', benchmark)
    os.makedirs(output_dir, exist_ok=True)
    
    for (table_name,) in table_list:
        select_query = f'SELECT * FROM "{table_name}"'
        conn = connector.get_connection()
        with conn.cursor() as cur:
            cur.execute(select_query)
            rows = cur.fetchall()
            column_names = [desc[0] for desc in cur.description]
        
        connector.close_connection(conn)
        csv_path = os.path.join(output_dir, f'{table_name}.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(column_names)
            writer.writerows(rows)
    
    connector.close_all_connection()


def clean_csv(benchmark):
    """
    清理指定benchmark下所有CSV文件中的数值列空值。
    - 删除标准: 任何数值列中有NULL或空值的行都会被删除
    """
    import pandas as pd
    import numpy as np
    import pickle
    
    # 获取数据集目录
    dataset_dir = os.path.join(current_path, '../dataset', benchmark)
    # 加载元数据以识别数值列
    metadata_path = os.path.join(dataset_dir, 'metadata.pkl')
    if not os.path.exists(metadata_path):
        print(f"错误: 元数据文件不存在 {metadata_path}")
        print("请先运行 --export_metadata 生成元数据")
        return
    
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    # 获取目录中的所有CSV文件
    csv_files = [f for f in os.listdir(dataset_dir) if f.endswith('.csv')]
    print(f"开始清理 {benchmark} 基准测试的CSV文件...")
    # 空值替换字典
    na_values = ["", "NULL", "null", "NA", "N/A", "None", "none"]
    for csv_file in csv_files:
        if csv_file=="catalog_sales.csv":
            continue
        
        # 移除过滤条件，处理所有CSV文件
        file_path = os.path.join(dataset_dir, csv_file)
        table_name = csv_file[:-4]  # 去掉.csv后缀
        # 检查元数据中是否有该表
        if table_name not in metadata:
            print(f"跳过: {csv_file} (未在元数据中找到)")
            continue
        print(f"正在处理: {csv_file}")
        try:
            # 使用pandas读取文件，处理各种空值
            df = pd.read_csv(file_path, na_values=na_values, keep_default_na=True, low_memory=False)
            # 获取该表的数值列
            numeric_columns = metadata[table_name]['numeric_columns']
            # 只考虑表中实际存在的数值列（避免元数据与CSV不一致的问题）
            numeric_columns = [col for col in numeric_columns if col in df.columns]
            if not numeric_columns:
                print(f"  - 警告: {csv_file} 中没有找到数值列，跳过清理")
                continue
            total_rows = len(df)
            print(f"  - 总行数: {total_rows}")
            print(f"  - 数值列: {', '.join(numeric_columns)}")
            # 统计数值列的空值情况
            null_counts = df[numeric_columns].isnull().sum()
            print("  - 各数值列的空值数:")
            for col, count in null_counts.items():
                if count > 0:
                    print(f"    * {col}: {count} 行有空值 ({count/total_rows:.2%})")
            # 有数值列空值的行总数
            rows_with_numeric_nulls = df[numeric_columns].isnull().any(axis=1).sum()
            print(f"  - 有数值列空值的行数: {rows_with_numeric_nulls} ({rows_with_numeric_nulls/total_rows:.2%})")
            if rows_with_numeric_nulls == 0:
                print("  - 没有数值列空值的行，文件保持不变")
                continue
            # 删除任何数值列有空值的行
            df_cleaned = df.dropna(subset=numeric_columns)
            # 保存清理后的文件，不包含索引
            df_cleaned.to_csv(file_path, index=False)
            # 统计清理结果
            removed_rows = len(df) - len(df_cleaned)
            print(f"  - 已清理 {removed_rows} 行数据 ({removed_rows/total_rows:.2%})")
            
        except Exception as e:
            print(f"  - 处理 {csv_file} 时出错: {str(e)}")
    
    print("CSV文件清理完成。")
    
    
    

# 输入参数benchmark
parser = argparse.ArgumentParser()
parser.add_argument('--benchmark', help='benchmark name', type=str, default='tpcds')
# 导出元数据命令  --export_metadata即可执行

parser.add_argument('--export_metadata', help='export metadata', action='store_true')
# 导出表格数据命令 --export_tables即可执行
parser.add_argument('--export_csv', help='export tables', action='store_true')
parser.add_argument('--clean_csv', help='export tables', action='store_true')

args = parser.parse_args()
args.clean_csv = True

if args.export_metadata:
    export_metadata(args.benchmark)
if args.export_csv:
    export_tables_to_csv(args.benchmark)
if args.clean_csv:
    clean_csv(args.benchmark)

