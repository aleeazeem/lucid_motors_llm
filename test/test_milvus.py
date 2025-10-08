from pymilvus import connections, utility

# Test connection
try:
    connections.connect("default", host="localhost", port="19530")
    print("✓ Connected to Milvus successfully!")
    print(f"Milvus version: {utility.get_server_version()}")
    connections.disconnect("default")
except Exception as e:
    print(f"❌ Connection failed: {e}")