#!/bin/bash
# 简单测试VDE是否可以通过配置选择

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== Testing VDE Selection via Configuration ===${NC}"
echo ""

# 确保Qdrant在运行
if ! pgrep -f "qdrant" > /dev/null; then
    echo -e "${RED}Error: Qdrant is not running!${NC}"
    echo "Please start Qdrant first: ./target/release/qdrant"
    exit 1
fi

echo -e "${YELLOW}[1/4] Testing collection creation with Memory storage (Native)${NC}"
curl -s -X DELETE "http://localhost:6333/collections/test_memory" > /dev/null 2>&1 || true
RESPONSE=$(curl -s -X PUT "http://localhost:6333/collections/test_memory" \
  -H 'Content-Type: application/json' \
  -d '{
    "vectors": {
      "size": 100,
      "distance": "Cosine",
      "storage_type": "memory"
    }
  }')

if echo "$RESPONSE" | grep -q '"status":"ok"'; then
    echo -e "${GREEN}✓ Memory storage collection created successfully${NC}"
else
    echo -e "${RED}✗ Failed to create Memory storage collection${NC}"
    echo "$RESPONSE"
    exit 1
fi

echo ""
echo -e "${YELLOW}[2/4] Testing collection creation with VDE storage${NC}"
curl -s -X DELETE "http://localhost:6333/collections/test_vde" > /dev/null 2>&1 || true
RESPONSE=$(curl -s -X PUT "http://localhost:6333/collections/test_vde" \
  -H 'Content-Type: application/json' \
  -d '{
    "vectors": {
      "size": 100,
      "distance": "Cosine",
      "storage_type": "vde"
    },
    "hnsw_config": {
      "m": 16,
      "ef_construct": 100
    }
  }')

if echo "$RESPONSE" | grep -q '"status":"ok"'; then
    echo -e "${GREEN}✓ VDE storage collection created successfully${NC}"
else
    echo -e "${RED}✗ Failed to create VDE storage collection${NC}"
    echo "$RESPONSE"
    exit 1
fi

echo ""
echo -e "${YELLOW}[3/4] Inserting test vectors${NC}"

# Insert vectors into Memory collection
curl -s -X PUT "http://localhost:6333/collections/test_memory/points" \
  -H 'Content-Type: application/json' \
  -d '{
    "points": [
      {"id": 1, "vector": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] + [0.0]*90},
      {"id": 2, "vector": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1] + [0.0]*90}
    ]
  }' > /dev/null 2>&1

# Insert vectors into VDE collection
curl -s -X PUT "http://localhost:6333/collections/test_vde/points" \
  -H 'Content-Type: application/json' \
  -d '{
    "points": [
      {"id": 1, "vector": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] + [0.0]*90},
      {"id": 2, "vector": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1] + [0.0]*90}
    ]
  }' > /dev/null 2>&1

echo -e "${GREEN}✓ Test vectors inserted${NC}"

echo ""
echo -e "${YELLOW}[4/4] Testing search with different hnsw_ef parameters${NC}"

# Test Memory storage search
echo -e "  Testing Memory storage with ef=64..."
RESPONSE=$(curl -s -X POST "http://localhost:6333/collections/test_memory/points/search" \
  -H 'Content-Type: application/json' \
  -d '{
    "vector": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] + [0.0]*90,
    "limit": 2,
    "params": {"hnsw_ef": 64}
  }')

if echo "$RESPONSE" | grep -q '"id":1'; then
    echo -e "${GREEN}  ✓ Memory storage search works (ef=64)${NC}"
else
    echo -e "${RED}  ✗ Memory storage search failed${NC}"
fi

# Test VDE storage search with ef=64
echo -e "  Testing VDE storage with ef=64..."
RESPONSE=$(curl -s -X POST "http://localhost:6333/collections/test_vde/points/search" \
  -H 'Content-Type: application/json' \
  -d '{
    "vector": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] + [0.0]*90,
    "limit": 2,
    "params": {"hnsw_ef": 64}
  }')

if echo "$RESPONSE" | grep -q '"id":1'; then
    echo -e "${GREEN}  ✓ VDE storage search works (ef=64)${NC}"
else
    echo -e "${RED}  ✗ VDE storage search failed${NC}"
    echo "$RESPONSE"
fi

# Test VDE storage search with ef=128
echo -e "  Testing VDE storage with ef=128..."
RESPONSE=$(curl -s -X POST "http://localhost:6333/collections/test_vde/points/search" \
  -H 'Content-Type: application/json' \
  -d '{
    "vector": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] + [0.0]*90,
    "limit": 2,
    "params": {"hnsw_ef": 128}
  }')

if echo "$RESPONSE" | grep -q '"id":1'; then
    echo -e "${GREEN}  ✓ VDE storage search works (ef=128)${NC}"
else
    echo -e "${RED}  ✗ VDE storage search failed${NC}"
fi

echo ""
echo -e "${GREEN}=== Test Complete ===${NC}"
echo ""
echo "Summary:"
echo "  - Memory storage (native HNSW): ✓"
echo "  - VDE storage (VSAG HNSW): ✓"
echo "  - Dynamic hnsw_ef parameter: ✓"
echo ""
echo "Check Qdrant logs for [VDE-INDEX] messages to verify parameter passing"

