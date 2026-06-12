import sqlite3

def analyze_db():
    conn = sqlite3.connect('active_learning.db')
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [t[0] for t in cursor.fetchall()]
    
    print("=== Tables in active_learning.db ===")
    print(", ".join(tables))
    print("\n" + "="*40 + "\n")
    
    for table in tables:
        print(f"Table: {table}")
        
        # Schema
        cursor.execute(f"PRAGMA table_info({table});")
        schema = cursor.fetchall()
        print("Schema:")
        for col in schema:
            print(f"  {col[1]} ({col[2]})")
            
        # Count
        cursor.execute(f"SELECT COUNT(*) FROM {table};")
        count = cursor.fetchone()[0]
        print(f"Row count: {count}")
        
        # Sample
        cursor.execute(f"SELECT * FROM {table} LIMIT 3;")
        sample = cursor.fetchall()
        print("Sample data:")
        for row in sample:
            print(f"  {row}")
            
        print("-" * 40)
        
    conn.close()

if __name__ == '__main__':
    analyze_db()
