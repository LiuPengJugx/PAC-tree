synetic_queries = {
        'Query 1': {
            'ranges': {
                'lineitem': {
                    'l_orderkey': {'min': 1, 'max': 6000000},
                    'l_partkey': {'min': 1, 'max': 200000},
                    'l_suppkey': {'min': 1, 'max': 10000},
                    'l_linenumber': {'min': 1, 'max': 7},
                    'l_quantity': {'min': 1.00, 'max': 50.00},
                    'l_extendedprice': {'min': 901.00, 'max': 104949.50},
                    'l_discount': {'min': 0.00, 'max': 0.10},
                    'l_tax': {'min': 0.00, 'max': 0.08},
                    'l_shipdate': {'min': '1992-01-02', 'max': '1998-10-01'},
                    'l_commitdate': {'min': '1992-01-31', 'max': '1998-10-31'},
                    'l_receiptdate': {'min': '1992-01-04', 'max': '1998-12-31'}
                }
            },
            'join_info': []
        },
        'Query 2': {
            'ranges': {
                'part': {
                    'p_partkey': {'min': 354, 'max': 199875},
                    'p_size': {'min': 38, 'max': 38},
                    'p_retailprice': {'min': 911.00, 'max': 2086.99}
                },
                'region': {
                    'r_regionkey': {'min': 2, 'max': 2}
                },
                'nation': {
                    'n_nationkey': {'min': 1, 'max': 24},
                    'n_regionkey': {'min': 2, 'max': 2}
                },
                'partsupp': {
                    'ps_partkey': {'min': 354, 'max': 199875},
                    'ps_suppkey': {'min': 1, 'max': 10000},
                    'ps_availqty': {'min': 1, 'max': 9999},
                    'ps_supplycost': {'min': 1.00, 'max': 1000.00}
                }
            },
            'join_info': [
                {
                    'join_type': 'Hash Join',
                    'join_keys': ['part.p_partkey=partsupp.ps_partkey']
                },
                {
                    'join_type': 'Hash Join',
                    'join_keys': ['nation.n_regionkey=region.r_regionkey']
                }
            ]
        },
        'Query 3': {
            'ranges': {
                'customer': {
                    'c_custkey': {'min': 9, 'max': 149983},
                    'c_nationkey': {'min': 0, 'max': 24},
                    'c_acctbal': {'min': -999.99, 'max': 9999.74}
                },
                'orders': {
                    'o_orderkey': {'min': 3, 'max': 5999975},
                    'o_custkey': {'min': 1, 'max': 149999},
                    'o_totalprice': {'min': 866.90, 'max': 555285.16},
                    'o_orderdate': {'min': '1992-01-01', 'max': '1995-03-16'},
                    'o_shippriority': {'min': 0, 'max': 0}
                },
                'lineitem': {
                    'l_orderkey': {'min': 1, 'max': 6000000},
                    'l_partkey': {'min': 1, 'max': 200000},
                    'l_suppkey': {'min': 1, 'max': 10000},
                    'l_linenumber': {'min': 1, 'max': 7},
                    'l_quantity': {'min': 1.00, 'max': 50.00},
                    'l_extendedprice': {'min': 901.00, 'max': 104949.50},
                    'l_discount': {'min': 0.00, 'max': 0.10},
                    'l_tax': {'min': 0.00, 'max': 0.08},
                    'l_shipdate': {'min': '1995-03-19', 'max': '1998-12-01'},
                    'l_commitdate': {'min': '1992-01-31', 'max': '1998-10-31'},
                    'l_receiptdate': {'min': '1992-01-04', 'max': '1998-12-31'}
                }
            },
            'join_info': [
                {
                    'join_type': 'Hash Join',
                    'join_keys': ['orders.o_custkey=customer.c_custkey']
                },
                {
                    'join_type': 'Hash Join',
                    'join_keys': ['orders.o_orderkey=lineitem.l_orderkey']
                }
            ]
        },
        'Query 4': {
            'ranges': {
                'orders': {
                    'o_orderkey': {'min': 32, 'max': 5999558},
                    'o_custkey': {'min': 5, 'max': 149999},
                    'o_totalprice': {'min': 895.39, 'max': 502906.33},
                    'o_orderdate': {'min': '1995-07-01', 'max': '1995-09-30'},
                    'o_shippriority': {'min': 0, 'max': 0}
                },
                'lineitem': {
                    'l_orderkey': {'min': 32, 'max': 5999558},
                    'l_partkey': {'min': 1, 'max': 200000},
                    'l_suppkey': {'min': 1, 'max': 10000},
                    'l_linenumber': {'min': 1, 'max': 7},
                    'l_quantity': {'min': 1.00, 'max': 50.00},
                    'l_extendedprice': {'min': 901.00, 'max': 104949.50},
                    'l_discount': {'min': 0.00, 'max': 0.10},
                    'l_tax': {'min': 0.00, 'max': 0.08},
                    'l_shipdate': {'min': '1992-01-06', 'max': '1998-12-01'},
                    'l_commitdate': {'min': '1992-01-31', 'max': '1998-10-31'},
                    'l_receiptdate': {'min': '1992-02-02', 'max': '1998-12-31'}
                }
            },
            'join_info': [
                {'join_type': 'Hash Join',
                 'join_keys': ['orders.o_orderkey=lineitem.l_orderkey']}
            ]
        },
        'Query 5': {
            'ranges': {
                'customer': {
                    'c_custkey': {'min': 4, 'max': 149999},
                    'c_nationkey': {'min': 0, 'max': 24},
                    'c_acctbal': {'min': -999.99, 'max': 9999.74}
                },
                'orders': {
                    'o_orderkey': {'min': 32, 'max': 5999558},
                    'o_custkey': {'min': 4, 'max': 149999},
                    'o_totalprice': {'min': 895.39, 'max': 510061.60},
                    'o_orderdate': {'min': '1993-01-01', 'max': '1993-12-31'},
                    'o_shippriority': {'min': 0, 'max': 0}
                },
                'lineitem': {
                    'l_orderkey': {'min': 1, 'max': 6000000},
                    'l_partkey': {'min': 1, 'max': 200000},
                    'l_suppkey': {'min': 1, 'max': 10000},
                    'l_linenumber': {'min': 1, 'max': 7},
                    'l_quantity': {'min': 1.00, 'max': 50.00},
                    'l_extendedprice': {'min': 901.00, 'max': 104949.50},
                    'l_discount': {'min': 0.00, 'max': 0.10},
                    'l_tax': {'min': 0.00, 'max': 0.08},
                    'l_shipdate': {'min': '1992-01-06', 'max': '1998-12-01'},
                    'l_commitdate': {'min': '1992-01-31', 'max': '1998-10-31'},
                    'l_receiptdate': {'min': '1992-02-02', 'max': '1998-12-31'}
                },
                'supplier': {
                    's_suppkey': {'min': 1, 'max': 10000},
                    's_nationkey': {'min': 0, 'max': 24},
                    's_acctbal': {'min': -998.22, 'max': 9999.72}
                },
                'nation': {
                    'n_nationkey': {'min': 0, 'max': 24},
                    'n_regionkey': {'min': 0, 'max': 4}
                },
                'region': {
                    'r_regionkey': {'min': 0, 'max': 4}
                }
            },
            'join_info': [
                {'join_type': 'Hash Join',
                 'join_keys': ['customer.c_custkey=orders.o_custkey']},
                {'join_type': 'Hash Join',
                 'join_keys': ['lineitem.l_orderkey=orders.o_orderkey']},
                {'join_type': 'Hash Join',
                 'join_keys': ['supplier.s_suppkey=lineitem.l_suppkey']},
                {'join_type': 'Hash Join',
                 'join_keys': ['supplier.s_nationkey=nation.n_nationkey']},
                {'join_type': 'Hash Join',
                 'join_keys': ['nation.n_regionkey=region.r_regionkey']}
            ]
        },
        'Query 6': {
            'ranges': {
                'lineitem': {
                    'l_orderkey': {'min': 1, 'max': 6000000},
                    'l_partkey': {'min': 1, 'max': 200000},
                    'l_suppkey': {'min': 1, 'max': 10000},
                    'l_linenumber': {'min': 1, 'max': 7},
                    'l_quantity': {'min': 1.00, 'max': 25.00},
                    'l_extendedprice': {'min': 901.00, 'max': 104949.50},
                    'l_discount': {'min': 0.06, 'max': 0.08},
                    'l_tax': {'min': 0.00, 'max': 0.08},
                    'l_shipdate': {'min': '1993-01-01', 'max': '1994-01-01'},
                    'l_commitdate': {'min': '1992-01-31', 'max': '1998-10-31'},
                    'l_receiptdate': {'min': '1992-02-02', 'max': '1998-12-31'}
                }
            },
            'join_info': []
        },
        'Query 7': {
            'ranges': {
                'nation': {
                    'n_nationkey': {'min': 16, 'max': 23},
                    'n_regionkey': {'min': 0, 'max': 3}
                },
                'lineitem': {
                    'l_orderkey': {'min': 1, 'max': 6000000},
                    'l_partkey': {'min': 1, 'max': 200000},
                    'l_suppkey': {'min': 1, 'max': 10000},
                    'l_linenumber': {'min': 1, 'max': 7},
                    'l_quantity': {'min': 1.00, 'max': 50.00},
                    'l_extendedprice': {'min': 904.00, 'max': 104749.50},
                    'l_discount': {'min': 0.00, 'max': 0.10},
                    'l_tax': {'min': 0.00, 'max': 0.08},
                    'l_shipdate': {'min': '1995-01-01', 'max': '1996-12-31'},
                    'l_commitdate': {'min': '1994-10-02', 'max': '1997-03-28'},
                    'l_receiptdate': {'min': '1995-01-02', 'max': '1997-01-30'}
                },
                'supplier': {
                    's_suppkey': {'min': 1, 'max': 10000},
                    's_nationkey': {'min': 16, 'max': 23},
                    'c_acctbal': {'min': -999.99, 'max': 9999.74}
                },
                'customer': {
                    'c_custkey': {'min': 9, 'max': 149983},
                    'c_nationkey': {'min': 16, 'max': 23},
                    'c_acctbal': {'min': -999.99, 'max': 9999.74}
                },
                'orders': {
                    'o_orderkey': {'min': 1, 'max': 6000000},
                    'o_custkey': {'min': 1, 'max': 149999},
                    'o_totalprice': {'min': 857.71, 'max': 555285.16},
                    'o_orderdate': {'min': '1992-01-01', 'max': '1998-08-02'},
                    'o_shippriority': {'min': 0, 'max': 0}
                }
                
            },
            'join_info': [
                {'join_type': 'Hash Join',
                 'join_keys': ['lineitem.l_suppkey=supplier.s_suppkey']},
                {'join_type': 'Hash Join',
                 'join_keys': ['customer.c_nationkey=nation.n_nationkey']},
                {'join_type': 'Hash Join',
                 'join_keys': ['supplier.s_nationkey=nation.n_nationkey']}
            ]
        },
        'Query 8': {
            'ranges': { 
                'part': {
                    'p_partkey': {'min': 113, 'max': 199995},
                },
                'orders': {
                    'o_orderdate': {'min': '1995-01-01', 'max': '1996-12-31'},   
                }
            },
            'join_info': []  
            
        },
        'Query 9': {
            'ranges': { 
            },
            'join_info': []  
            
        },
        'Query 10': {
            'ranges': { 
            },
            'join_info': []  
            
        },
        'Query 11': {
            'ranges': { 
            },
            'join_info': []  
            
        },
        'Query 12': {
            'ranges': { 
            },
            'join_info': []  
            
        },
        'Query 13': {
            'ranges': { 
            },
            'join_info': []  
            
        },
        'Query 14': {
            'ranges': { 
            },
            'join_info': []  
            
        },
        'Query 15': {
            'ranges': { 
            },
            'join_info': []  
            
        },
        'Query 16': {
            'ranges': { 
            },
            'join_info': []  
            
        },
        'Query 17': {
            'ranges': { 
            },
            'join_info': []  
            
        },
        'Query 18': {
            'ranges': { 
            },
            'join_info': []  
            
        },
        'Query 19': {
            'ranges': { 
            },
            'join_info': []  
            
        },
        'Query 20': {
            'ranges': { 
            },
            'join_info': []  
            
        },
        'Query 21': {
            'ranges': { 
            },
            'join_info': []  
            
        },
        
    }