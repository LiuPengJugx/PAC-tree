import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *

# Set Python environment
os.environ['PYSPARK_PYTHON'] = sys.executable

def create_spark_session():
    """Create Spark session"""
    spark = SparkSession.builder \
        .appName("ComprehensiveSparkTest") \
        .master("local[*]") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    return spark

def create_sample_tables(spark):
    """Create sample tables and data"""
    
    # 1. Create customers table
    customers_schema = StructType([
        StructField("customer_id", IntegerType(), False),
        StructField("name", StringType(), True),
        StructField("email", StringType(), True),
        StructField("city", StringType(), True),
        StructField("registration_date", DateType(), True)
    ])
    
    customers_data = [
        (1, "John Smith", "john.smith@email.com", "New York", "2023-01-15"),
        (2, "Alice Johnson", "alice.johnson@email.com", "Los Angeles", "2023-02-20"),
        (3, "Bob Wilson", "bob.wilson@email.com", "Chicago", "2023-03-10"),
        (4, "Carol Davis", "carol.davis@email.com", "Houston", "2023-04-05"),
        (5, "David Miller", "david.miller@email.com", "Phoenix", "2023-05-12")
    ]
    
    customers_df = spark.createDataFrame(customers_data, customers_schema)
    customers_df.createOrReplaceTempView("customers")
    
    # 2. Create orders table
    orders_schema = StructType([
        StructField("order_id", IntegerType(), False),
        StructField("customer_id", IntegerType(), True),
        StructField("product_id", IntegerType(), True),
        StructField("quantity", IntegerType(), True),
        StructField("price", DecimalType(10, 2), True),
        StructField("order_date", DateType(), True)
    ])
    
    orders_data = [
        (1001, 1, 101, 2, 299.99, "2023-06-01"),
        (1002, 2, 102, 1, 199.99, "2023-06-02"),
        (1003, 1, 103, 3, 99.99, "2023-06-03"),
        (1004, 3, 104, 1, 499.99, "2023-06-04"),
        (1005, 4, 101, 2, 299.99, "2023-06-05"),
        (1006, 2, 105, 1, 149.99, "2023-06-06"),
        (1007, 5, 102, 2, 199.99, "2023-06-07")
    ]
    
    orders_df = spark.createDataFrame(orders_data, orders_schema)
    orders_df.createOrReplaceTempView("orders")
    
    # 3. Create products table
    products_schema = StructType([
        StructField("product_id", IntegerType(), False),
        StructField("product_name", StringType(), True),
        StructField("category", StringType(), True),
        StructField("price", DecimalType(10, 2), True),
        StructField("stock_quantity", IntegerType(), True)
    ])
    
    products_data = [
        (101, "iPhone 15", "Electronics", 299.99, 100),
        (102, "MacBook Air", "Electronics", 199.99, 50),
        (103, "AirPods", "Electronics", 99.99, 200),
        (104, "iPad Pro", "Electronics", 499.99, 75),
        (105, "Apple Watch", "Electronics", 149.99, 150)
    ]
    
    products_df = spark.createDataFrame(products_data, products_schema)
    products_df.createOrReplaceTempView("products")
    
    return customers_df, orders_df, products_df

def run_single_table_queries(spark):
    """Execute single table queries"""
    print("=== Single Table Query Results ===")
    
    # 1. Query all customers
    print("1. All customer information:")
    spark.sql("SELECT * FROM customers ORDER BY customer_id").show()
    
    # 2. Query high-value orders
    print("2. Orders with amount > 200:")
    spark.sql("SELECT * FROM orders WHERE price > 200 ORDER BY price DESC").show()
    
    # 3. Product category statistics
    print("3. Product category statistics:")
    spark.sql("SELECT category, COUNT(*) as count FROM products GROUP BY category").show()
    
    # 4. Query low stock products
    print("4. Products with low stock (< 100):")
    spark.sql("SELECT product_name, stock_quantity FROM products WHERE stock_quantity < 100").show()

def run_multi_table_queries(spark):
    """Execute multi-table join queries"""
    print("=== Multi-Table Query Results ===")
    
    # 1. Query customer order details
    print("1. Customer order details:")
    customer_orders_sql = """
        SELECT 
            c.name,
            c.email,
            o.order_id,
            o.quantity,
            o.price,
            o.order_date
        FROM customers c
        JOIN orders o ON c.customer_id = o.customer_id
        ORDER BY o.order_date
    """
    spark.sql(customer_orders_sql).show()
    
    # 2. Query order product details
    print("2. Order product details:")
    order_products_sql = """
        SELECT 
            o.order_id,
            c.name as customer_name,
            p.product_name,
            p.category,
            o.quantity,
            o.price,
            (o.quantity * o.price) as total_amount
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        JOIN products p ON o.product_id = p.product_id
        ORDER BY o.order_id
    """
    spark.sql(order_products_sql).show()
    
    # 3. Customer spending statistics
    print("3. Customer spending statistics:")
    customer_spending_sql = """
        SELECT 
            c.name as customer_name,
            c.city,
            COUNT(o.order_id) as order_count,
            SUM(o.quantity * o.price) as total_spent,
            AVG(o.quantity * o.price) as avg_order_value
        FROM customers c
        JOIN orders o ON c.customer_id = o.customer_id
        GROUP BY c.customer_id, c.name, c.city
        ORDER BY total_spent DESC
    """
    spark.sql(customer_spending_sql).show()
    
    # 4. Query popular products
    print("4. Popular product sales statistics:")
    popular_products_sql = """
        SELECT 
            p.product_name,
            p.category,
            SUM(o.quantity) as total_sold,
            SUM(o.quantity * o.price) as total_revenue,
            COUNT(DISTINCT o.customer_id) as unique_customers
        FROM products p
        JOIN orders o ON p.product_id = o.product_id
        GROUP BY p.product_id, p.product_name, p.category
        ORDER BY total_sold DESC
    """
    spark.sql(popular_products_sql).show()

def run_advanced_queries(spark):
    """Execute advanced queries"""
    print("=== Advanced Query Results ===")
    
    # 1. Window function query
    print("1. Customer order ranking:")
    window_sql = """
        SELECT 
            name,
            order_id,
            total_amount,
            RANK() OVER (PARTITION BY name ORDER BY total_amount DESC) as order_rank
        FROM (
            SELECT 
                c.name,
                o.order_id,
                (o.quantity * o.price) as total_amount
            FROM customers c
            JOIN orders o ON c.customer_id = o.customer_id
        ) order_details
    """
    spark.sql(window_sql).show()
    
    # 2. Subquery
    print("2. High-value customer product preferences:")
    subquery_sql = """
        SELECT 
            p.category,
            COUNT(*) as order_count,
            AVG(o.quantity * o.price) as avg_order_value
        FROM orders o
        JOIN products p ON o.product_id = p.product_id
        WHERE o.customer_id IN (
            SELECT customer_id 
            FROM orders 
            GROUP BY customer_id 
            HAVING SUM(quantity * price) > 500
        )
        GROUP BY p.category
        ORDER BY avg_order_value DESC
    """
    spark.sql(subquery_sql).show()

def main():
    """Main function"""
    try:
        # Create Spark session
        spark = create_spark_session()
        print("Spark session created successfully!")
        
        # Create tables and data
        customers_df, orders_df, products_df = create_sample_tables(spark)
        print("Tables and data created successfully!")
        
        # Display table schemas
        print("=== Table Schemas ===")
        print("Customers table:")
        customers_df.printSchema()
        print("Orders table:")
        orders_df.printSchema()
        print("Products table:")
        products_df.printSchema()
        
        # Execute queries
        run_single_table_queries(spark)
        run_multi_table_queries(spark)
        run_advanced_queries(spark)
        
        # Statistics
        print("=== Data Overview ===")
        print(f"Total customers: {spark.sql('SELECT COUNT(*) FROM customers').collect()[0][0]}")
        print(f"Total orders: {spark.sql('SELECT COUNT(*) FROM orders').collect()[0][0]}")
        print(f"Total products: {spark.sql('SELECT COUNT(*) FROM products').collect()[0][0]}")
        
    except Exception as e:
        print(f"Error occurred during execution: {str(e)}")
    finally:
        # Close Spark session
        spark.stop()
        print("Spark session closed")

if __name__ == "__main__":
    main()