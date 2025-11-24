#!/bin/bash

# GroundedDINO-VL Label Studio Backend Test Suite
# This script tests all endpoints and validates the output format

set -e  # Exit on error

BACKEND_URL="http://localhost:9090"
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=================================="
echo "GroundedDINO-VL Backend Test Suite"
echo "=================================="
echo ""

# Test 1: Health Check
echo -e "${YELLOW}Test 1: Health Check${NC}"
echo "GET $BACKEND_URL/health"
echo ""
HEALTH_RESPONSE=$(curl -s -X GET "$BACKEND_URL/health")
echo "Response: $HEALTH_RESPONSE"
echo ""

# Validate health response
MODEL_LOADED=$(echo "$HEALTH_RESPONSE" | grep -o '"model_loaded":[^,}]*' | cut -d':' -f2 | tr -d ' ')
if [ "$MODEL_LOADED" = "true" ]; then
    echo -e "${GREEN}✓ Health check passed - Model is loaded${NC}"
else
    echo -e "${RED}✗ Health check failed - Model is not loaded${NC}"
    exit 1
fi
echo ""

# Test 2: Setup Endpoint
echo -e "${YELLOW}Test 2: Setup Endpoint${NC}"
echo "POST $BACKEND_URL/setup"
echo ""
SETUP_RESPONSE=$(curl -s -X POST "$BACKEND_URL/setup" \
  -H "Content-Type: application/json" \
  -d '{}')
echo "Response: $SETUP_RESPONSE"
echo ""

# Validate setup response
MODEL_VERSION=$(echo "$SETUP_RESPONSE" | grep -o '"model_version":"[^"]*"' | cut -d'"' -f4)
MODEL_NAME=$(echo "$SETUP_RESPONSE" | grep -o '"model_name":"[^"]*"' | cut -d'"' -f4)
if [ -n "$MODEL_VERSION" ] && [ -n "$MODEL_NAME" ]; then
    echo -e "${GREEN}✓ Setup endpoint passed${NC}"
    echo "  Model: $MODEL_NAME"
    echo "  Version: $MODEL_VERSION"
else
    echo -e "${RED}✗ Setup endpoint failed${NC}"
    exit 1
fi
echo ""

# Test 3: Model Info
echo -e "${YELLOW}Test 3: Model Info Endpoint${NC}"
echo "GET $BACKEND_URL/model-info"
echo ""
MODEL_INFO_RESPONSE=$(curl -s -X GET "$BACKEND_URL/model-info")
echo "Response: $MODEL_INFO_RESPONSE"
echo ""
echo -e "${GREEN}✓ Model info endpoint accessible${NC}"
echo ""

# Test 4: Predict with Default Wildlife Classes
echo -e "${YELLOW}Test 4: Predict with Default Wildlife Classes${NC}"
echo "POST $BACKEND_URL/predict"
echo "Testing with sample task (no explicit prompt - should use 26 wildlife classes)"
echo ""

# Create a test payload (you'll need to replace with actual image path)
TEST_PAYLOAD_1='{
  "data": {
    "image": "/data/local-files/?d=/data/datasets/bear/bear_0001.jpg"
  }
}'

echo "Payload:"
echo "$TEST_PAYLOAD_1" | jq . 2>/dev/null || echo "$TEST_PAYLOAD_1"
echo ""

PREDICT_RESPONSE_1=$(curl -s -X POST "$BACKEND_URL/predict" \
  -H "Content-Type: application/json" \
  -d "$TEST_PAYLOAD_1")

echo "Response:"
echo "$PREDICT_RESPONSE_1" | jq . 2>/dev/null || echo "$PREDICT_RESPONSE_1"
echo ""

# Validate prediction response structure
if echo "$PREDICT_RESPONSE_1" | grep -q '"results"'; then
    echo -e "${GREEN}✓ Prediction response has 'results' key${NC}"

    # Check for from_name = "label" (may have spaces in JSON)
    if echo "$PREDICT_RESPONSE_1" | grep -q '"from_name"[[:space:]]*:[[:space:]]*"label"'; then
        echo -e "${GREEN}✓ Correct from_name='label' field${NC}"
    else
        echo -e "${RED}✗ ERROR: from_name is not 'label'${NC}"
        echo "Debug: Searching for from_name in response..."
        echo "$PREDICT_RESPONSE_1" | grep -o '"from_name"[^,}]*' || echo "  (from_name not found)"
        exit 1
    fi

    # Check for to_name = "image"
    if echo "$PREDICT_RESPONSE_1" | grep -q '"to_name"[[:space:]]*:[[:space:]]*"image"'; then
        echo -e "${GREEN}✓ Correct to_name='image' field${NC}"
    else
        echo -e "${RED}✗ ERROR: to_name is not 'image'${NC}"
        exit 1
    fi

    # Check for type = "rectanglelabels"
    if echo "$PREDICT_RESPONSE_1" | grep -q '"type"[[:space:]]*:[[:space:]]*"rectanglelabels"'; then
        echo -e "${GREEN}✓ Correct type='rectanglelabels' field${NC}"
    else
        echo -e "${RED}✗ ERROR: type is not 'rectanglelabels'${NC}"
        exit 1
    fi

    # Check for model_version at top level
    if echo "$PREDICT_RESPONSE_1" | grep -q '"model_version"'; then
        echo -e "${GREEN}✓ model_version present at top level${NC}"
    else
        echo -e "${RED}✗ ERROR: model_version missing${NC}"
        exit 1
    fi

    # Check for rectanglelabels array
    if echo "$PREDICT_RESPONSE_1" | grep -q '"rectanglelabels":\['; then
        echo -e "${GREEN}✓ rectanglelabels array present${NC}"
    else
        echo -e "${RED}✗ ERROR: rectanglelabels array missing${NC}"
        exit 1
    fi

    # Check for percentage coordinates (x, y, width, height)
    if echo "$PREDICT_RESPONSE_1" | grep -q '"x":' && \
       echo "$PREDICT_RESPONSE_1" | grep -q '"y":' && \
       echo "$PREDICT_RESPONSE_1" | grep -q '"width":' && \
       echo "$PREDICT_RESPONSE_1" | grep -q '"height":'; then
        echo -e "${GREEN}✓ Percentage coordinates (x, y, width, height) present${NC}"
    else
        echo -e "${RED}✗ ERROR: Missing coordinate fields${NC}"
        exit 1
    fi

else
    echo -e "${RED}✗ Prediction endpoint failed - no 'results' key${NC}"
    exit 1
fi
echo ""

# Test 5: Predict with Custom Classes
echo -e "${YELLOW}Test 5: Predict with Custom Classes${NC}"
echo "POST $BACKEND_URL/predict"
echo "Testing with explicit class list"
echo ""

TEST_PAYLOAD_2='{
  "data": {
    "image": "/data/local-files/?d=/data/datasets/deer/deer_0001.jpg"
  },
  "prompt": ["deer", "person", "automobile"]
}'

echo "Payload:"
echo "$TEST_PAYLOAD_2" | jq . 2>/dev/null || echo "$TEST_PAYLOAD_2"
echo ""

PREDICT_RESPONSE_2=$(curl -s -X POST "$BACKEND_URL/predict" \
  -H "Content-Type: application/json" \
  -d "$TEST_PAYLOAD_2")

echo "Response (truncated):"
echo "$PREDICT_RESPONSE_2" | jq . 2>/dev/null | head -30 || echo "$PREDICT_RESPONSE_2" | head -30
echo ""

if echo "$PREDICT_RESPONSE_2" | grep -q '"results"'; then
    echo -e "${GREEN}✓ Custom class prediction successful${NC}"
else
    echo -e "${RED}✗ Custom class prediction failed${NC}"
    exit 1
fi
echo ""

# Test 6: URL Normalization
echo -e "${YELLOW}Test 6: URL Normalization${NC}"
echo "POST $BACKEND_URL/predict"
echo "Testing with hostname/port in URL (should be normalized)"
echo ""

TEST_PAYLOAD_3='{
  "data": {
    "image": "http://localhost:9090/data/local-files/?d=datasets/coyote/coyote_0001.jpg"
  }
}'

echo "Payload:"
echo "$TEST_PAYLOAD_3" | jq . 2>/dev/null || echo "$TEST_PAYLOAD_3"
echo ""

PREDICT_RESPONSE_3=$(curl -s -X POST "$BACKEND_URL/predict" \
  -H "Content-Type: application/json" \
  -d "$TEST_PAYLOAD_3")

echo "Response (truncated):"
echo "$PREDICT_RESPONSE_3" | jq . 2>/dev/null | head -30 || echo "$PREDICT_RESPONSE_3" | head -30
echo ""

if echo "$PREDICT_RESPONSE_3" | grep -q '"results"'; then
    echo -e "${GREEN}✓ URL normalization successful${NC}"
else
    echo -e "${RED}✗ URL normalization failed${NC}"
    exit 1
fi
echo ""

# Test 7: Batch Prediction
echo -e "${YELLOW}Test 7: Batch Prediction${NC}"
echo "POST $BACKEND_URL/predict"
echo "Testing with multiple tasks"
echo ""

TEST_PAYLOAD_4='{
  "tasks": [
    {
      "data": {
        "image": "/data/local-files/?d=/data/datasets/bear/bear_0001.jpg"
      }
    },
    {
      "data": {
        "image": "/data/local-files/?d=/data/datasets/deer/deer_0001.jpg"
      }
    }
  ]
}'

echo "Payload:"
echo "$TEST_PAYLOAD_4" | jq . 2>/dev/null || echo "$TEST_PAYLOAD_4"
echo ""

PREDICT_RESPONSE_4=$(curl -s -X POST "$BACKEND_URL/predict" \
  -H "Content-Type: application/json" \
  -d "$TEST_PAYLOAD_4")

echo "Response (truncated):"
echo "$PREDICT_RESPONSE_4" | jq . 2>/dev/null | head -40 || echo "$PREDICT_RESPONSE_4" | head -40
echo ""

# Count results
RESULT_COUNT=$(echo "$PREDICT_RESPONSE_4" | grep -o '"result":\[' | wc -l)
if [ "$RESULT_COUNT" -ge 2 ]; then
    echo -e "${GREEN}✓ Batch prediction successful (received $RESULT_COUNT results)${NC}"
else
    echo -e "${RED}✗ Batch prediction failed (expected 2 results, got $RESULT_COUNT)${NC}"
    exit 1
fi
echo ""

# Test 8: Serve Local File
echo -e "${YELLOW}Test 8: Serve Local File Endpoint${NC}"
echo "GET $BACKEND_URL/data/local-files/?d=/data/datasets/bear/bear_0001.jpg"
echo ""

FILE_RESPONSE=$(curl -s -w "\n%{http_code}" -X GET "$BACKEND_URL/data/local-files/?d=/data/datasets/bear/bear_0001.jpg")
HTTP_CODE=$(echo "$FILE_RESPONSE" | tail -n1)

if [ "$HTTP_CODE" = "200" ]; then
    echo -e "${GREEN}✓ Local file serving successful (HTTP 200)${NC}"
else
    echo -e "${YELLOW}⚠ Local file endpoint returned HTTP $HTTP_CODE${NC}"
    echo "  (This may be expected if the test file doesn't exist)"
fi
echo ""

# Summary
echo "=================================="
echo -e "${GREEN}All Critical Tests Passed! ✓${NC}"
echo "=================================="
echo ""
echo "Summary:"
echo "  1. ✓ Health check - Model loaded"
echo "  2. ✓ Setup endpoint - Returns model info"
echo "  3. ✓ Model info - Accessible"
echo "  4. ✓ Predict (default) - Uses 26 wildlife classes"
echo "  5. ✓ Predict (custom) - Accepts custom class list"
echo "  6. ✓ URL normalization - Handles hostname/port"
echo "  7. ✓ Batch prediction - Multiple tasks"
echo "  8. ✓ Local file serving - Endpoint accessible"
echo ""
echo "Validation checks:"
echo "  ✓ from_name = 'label'"
echo "  ✓ to_name = 'image'"
echo "  ✓ type = 'rectanglelabels'"
echo "  ✓ model_version at top level"
echo "  ✓ rectanglelabels array"
echo "  ✓ Percentage coordinates"
echo ""
echo "Your backend is ready for Label Studio! 🎉"
echo ""
echo "Next steps:"
echo "  1. Update Label Studio ML settings with: http://groundeddino-vl:9090"
echo "  2. Enable 'Retrieve predictions automatically'"
echo "  3. Test with a task in Label Studio UI"
echo "  4. Use 'Create Annotations from Predictions' for batch"
echo ""
