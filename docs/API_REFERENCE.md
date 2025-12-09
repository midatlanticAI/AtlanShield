# ATLAN API Reference

## Overview

ATLAN provides a REST API for LLM output validation. All endpoints require authentication via API key.

**Base URL**: `http://localhost:8000` (development) or your deployed URL

**Authentication**: Include `X-API-Key` header with all requests

---

## Endpoints

### Health Check

```
GET /health
```

Returns server health status. No authentication required.

**Response**:
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

---

### Resonate (Validate)

```
POST /atlan/resonate
```

Validates text against ingested truth/policy.

**Headers**:
```
Content-Type: application/json
X-API-Key: your-api-key
```

**Request Body**:
```json
{
  "text": "The interest rate is 12.99%",
  "check_type": "TRUTH_CHECK",
  "truth_signature": 880.0
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| text | string | Yes | Text to validate |
| check_type | string | No | Type of check (default: "TRUTH_CHECK") |
| truth_signature | float | No | Context frequency for harmonic filtering |

**Response**:
```json
{
  "resonance": {
    "status": "RESONANT",
    "match_score": 0.892,
    "harmonic_factor": 1.0,
    "final_score": 0.892,
    "global_dissonance": 0.0,
    "matched_phrase": "The interest rate is 12.99%",
    "reason": "The interest rate is 12.99%"
  }
}
```

**Status Values**:
- `RESONANT` - Text aligns with truth (ALLOW)
- `DISSONANT` - Text contradicts truth (BLOCK)
- `AMBIGUOUS` - Uncertain match (review recommended)

**Dissonance Reasons**:
- `"Foreign Concepts: quantum, blockchain"` - Unknown vocabulary
- `"Unknown Numbers: [500000.0] not in truth database"` - Unknown numbers
- `"Negation Mismatch: Input contains negation"` - Negation detected
- `"Number Mismatch: [13.0] vs [12.99]"` - Number doesn't match policy
- `"No matching policy found"` - No similar truth exists

---

### Ingest Policy/Truth

```
POST /api/ingest
```

Ingests truth documents into ATLAN's memory.

**Request Body**:
```json
{
  "content": "The maximum loan amount is $50,000. Interest rates range from 5.99% to 24.99% APR.",
  "source_name": "loan_policy_v2"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| content | string | Yes | Policy/truth text to ingest |
| source_name | string | Yes | Identifier for this content |

**Response**:
```json
{
  "status": "success",
  "added_nodes": 2
}
```

The content is automatically:
1. Split into sentences
2. Vectorized for semantic search
3. Added to vocabulary whitelist
4. Numbers extracted and added to number whitelist

---

### Analytics - System Status

```
GET /analytics/status
```

Returns real-time system health metrics.

**Response**:
```json
{
  "dissonance": 0.15,
  "drift": 0.02,
  "status": "Stable",
  "timestamp": 0
}
```

| Field | Description |
|-------|-------------|
| dissonance | Current system dissonance level (0-1) |
| drift | Truth drift since calibration |
| status | "Stable" if drift < 0.1, else "Unstable" |

---

### Analytics - Heatmap

```
GET /analytics/heatmap
```

Returns top dissonance hotspots for monitoring.

**Response**:
```json
[
  {"phrase": "interest rate", "dissonance": 0.3},
  {"phrase": "loan amount", "dissonance": 0.2}
]
```

---

### Admin - Save Memory

```
POST /admin/save
```

Manually triggers persistence save.

**Response**:
```json
{
  "status": "saved",
  "nodes": 150
}
```

---

## Error Responses

### 401 Unauthorized

```json
{
  "detail": "Could not validate credentials"
}
```

Missing or invalid API key.

### 403 Forbidden

```json
{
  "detail": "Insufficient permissions. Required: admin"
}
```

API key doesn't have required role.

### 422 Validation Error

```json
{
  "detail": [
    {
      "loc": ["body", "text"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

Missing or invalid request body fields.

---

## Integration Examples

### Python (requests)

```python
import requests

API_URL = "http://localhost:8000"
API_KEY = "your-api-key"

headers = {
    "Content-Type": "application/json",
    "X-API-Key": API_KEY
}

# Ingest policy
requests.post(
    f"{API_URL}/api/ingest",
    headers=headers,
    json={
        "content": "Maximum loan is $50,000.",
        "source_name": "policy"
    }
)

# Validate text
response = requests.post(
    f"{API_URL}/atlan/resonate",
    headers=headers,
    json={"text": "Maximum loan is $50,000."}
)

result = response.json()
if result["resonance"]["status"] == "DISSONANT":
    print(f"BLOCKED: {result['resonance']['reason']}")
else:
    print("ALLOWED")
```

### cURL

```bash
# Ingest policy
curl -X POST http://localhost:8000/api/ingest \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"content": "Maximum loan is $50,000.", "source_name": "policy"}'

# Validate text
curl -X POST http://localhost:8000/atlan/resonate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"text": "Maximum loan is $50,000."}'
```

### JavaScript (fetch)

```javascript
const API_URL = "http://localhost:8000";
const API_KEY = "your-api-key";

async function validateText(text) {
  const response = await fetch(`${API_URL}/atlan/resonate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": API_KEY
    },
    body: JSON.stringify({ text })
  });

  const result = await response.json();
  return result.resonance.status !== "DISSONANT";
}
```

### LangChain Integration

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# Custom ATLAN guardrail
class AtlanGuardrail:
    def __init__(self, url, api_key):
        self.url = url
        self.api_key = api_key

    def __call__(self, text):
        import requests
        response = requests.post(
            f"{self.url}/atlan/resonate",
            headers={"X-API-Key": self.api_key},
            json={"text": text}
        )
        result = response.json()
        if result["resonance"]["status"] == "DISSONANT":
            raise ValueError(f"Blocked: {result['resonance']['reason']}")
        return text

# Use in chain
llm = ChatOpenAI()
guardrail = AtlanGuardrail("http://localhost:8000", "your-key")

chain = llm | guardrail
```

---

## Rate Limits

| Tier | Requests/min | Burst |
|------|--------------|-------|
| Free | 60 | 10 |
| Pro | 600 | 100 |
| Enterprise | Unlimited | Unlimited |

---

## Changelog

### v1.0.0 (2024-12)
- Initial release
- Fail-closed vocabulary lock
- Fail-closed number verification
- Negation detection with semantic opposites
- Harmonic resonance scoring
- LangChain integration
