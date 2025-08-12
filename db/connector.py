# 创建一个postgreSQL数据库连接类
import psycopg2
from psycopg2 import pool
from psycopg2.extras import DictCursor
from psycopg2.extras import RealDictCursor

class Connector:
    
    def __init__(self,database_name):
        db_config = {
            'user': 'postgres',
            'password': 'postgres',
            'host': 'localhost',
            'port': '5432',
            'database': database_name
        }
        self.db_config = db_config
        self.pool = None

    def connect(self):
        self.pool = psycopg2.pool.SimpleConnectionPool(1, 20, **self.db_config)

    def get_connection(self):
        return self.pool.getconn()

    def close_connection(self, conn):
        self.pool.putconn(conn)

    def close_all_connection(self):
        self.pool.closeall()

    def execute_query(self, query, params=None, cursor_factory=None):
        conn = self.get_connection()
        cursor = conn.cursor(cursor_factory=cursor_factory)
        cursor.execute(query, params)
        conn.commit()
        result = cursor.fetchall()
        cursor.close()
        self.close_connection(conn)
        return result

    def execute_query_one(self, query, params=None, cursor_factory=None):
        conn = self.get_connection()
        cursor = conn.cursor(cursor_factory=cursor_factory)
        cursor.execute(query, params)
        conn.commit()
        result = cursor.fetchone()
        cursor.close()
        self.close_connection(conn)
        return result

    def execute_update(self, query, params=None):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        cursor.close()
        self.close_connection(conn)

    def execute_insert(self, query, params=None):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        cursor.close()
        self.close_connection(conn)

    def execute_delete(self, query, params=None):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        cursor.close()
        self.close_connection(conn)

    def execute_insert_returning(self, query, params=None):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        result = cursor.fetchone()
        cursor.close()
        self.close_connection(conn)
        return result

    def execute_insert_many(self, query, params=None):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.executemany(query, params)
        conn.commit()
        cursor.close()
        self.close_connection(conn)

    def execute_query_dict(self, query, params=None):
        return self.execute_query(query, params, cursor_factory=DictCursor)
