# ATLAN Quick Start Guide

Get ATLAN running in under 5 minutes.

## Installation

```bash
# Clone and install
cd commercialization
pip install -r requirements.txt

# Start the server
cd atlan_server
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

## Step 1: Ingest Your Policy

```bash
curl -X POST http://localhost:8000/api/ingest \
  -H "Content-Type: application/json" \
  -H "X-API-Key: atlan-secret-key-123" \
  -d '{
    "content": "The maximum loan amount is $50,000. Interest rates range from 5.99% to 12.99% APR.",
    "source_name": "loan_policy"
  }'
```

Response:
```json
{"status": "success", "added_nodes": 2}
```

## Step 2: Validate LLM Output

### Allow valid output:
```bash
curl -X POST http://localhost:8000/atlan/resonate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: atlan-secret-key-123" \
  -d '{"text": "The maximum loan amount is $50,000."}'
```

Response:
```json
{
  "resonance": {
    "status": "RESONANT",
    "match_score": 0.95,
    "reason": "The maximum loan amount is $50,000."
  }
}
```

### Block hallucinated numbers:
```bash
curl -X POST http://localhost:8000/atlan/resonate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: atlan-secret-key-123" \
  -d '{"text": "The maximum loan amount is $500,000."}'
```

Response:
```json
{
  "resonance": {
    "status": "DISSONANT",
    "reason": "Unknown Numbers: [500000.0] not in truth database"
  }
}
```

### Block negation attacks:
```bash
curl -X POST http://localhost:8000/atlan/resonate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: atlan-secret-key-123" \
  -d '{"text": "Insurance is not required."}'
```

Response:
```json
{
  "resonance": {
    "status": "DISSONANT",
    "reason": "Negation Mismatch: Input contains negation"
  }
}
```

### Block foreign vocabulary:
```bash
curl -X POST http://localhost:8000/atlan/resonate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: atlan-secret-key-123" \
  -d '{"text": "Quantum blockchain verification required."}'
```

Response:
```json
{
  "resonance": {
    "status": "DISSONANT",
    "reason": "Foreign Concepts: quantum, blockchain"
  }
}
```

## Python Integration

```python
import requests

class AtlanGuard:
    def __init__(self, url="http://localhost:8000", api_key="atlan-secret-key-123"):
        self.url = url
        self.headers = {"X-API-Key": api_key, "Content-Type": "application/json"}

    def ingest(self, content, source_name):
        resp = requests.post(
            f"{self.url}/api/ingest",
            headers=self.headers,
            json={"content": content, "source_name": source_name}
        )
        return resp.json()

    def check(self, text):
        resp = requests.post(
            f"{self.url}/atlan/resonate",
            headers=self.headers,
            json={"text": text}
        )
        result = resp.json()
        return result["resonance"]["status"] != "DISSONANT"

    def validate(self, text):
        """Returns (allowed: bool, reason: str)"""
        resp = requests.post(
            f"{self.url}/atlan/resonate",
            headers=self.headers,
            json={"text": text}
        )
        result = resp.json()["resonance"]
        return result["status"] != "DISSONANT", result.get("reason", "")

# Usage
guard = AtlanGuard()

# Ingest policy
guard.ingest("Maximum loan is $50,000 at 12.99% APR.", "policy")

# Quick check
if guard.check("Maximum loan is $50,000."):
    print("ALLOWED")
else:
    print("BLOCKED")

# Detailed validation
allowed, reason = guard.validate("Maximum loan is $500,000.")
print(f"Allowed: {allowed}, Reason: {reason}")
# Output: Allowed: False, Reason: Unknown Numbers: [500000.0] not in truth database
```

## LangChain Integration

```python
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

class AtlanOutputGuard:
    def __init__(self, url="http://localhost:8000", api_key="atlan-secret-key-123"):
        self.url = url
        self.api_key = api_key

    def __call__(self, text):
        import requests
        resp = requests.post(
            f"{self.url}/atlan/resonate",
            headers={"X-API-Key": self.api_key},
            json={"text": text}
        )
        result = resp.json()
        if result["resonance"]["status"] == "DISSONANT":
            raise ValueError(f"BLOCKED: {result['resonance']['reason']}")
        return text

# Create guardrailed chain
llm = ChatOpenAI(model="gpt-3.5-turbo")
guard = AtlanOutputGuard()

chain = llm | StrOutputParser() | guard

# Every response is validated before returning
try:
    result = chain.invoke("What is the maximum loan amount?")
    print(result)
except ValueError as e:
    print(f"Response blocked: {e}")
```

## Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/ingest` | POST | Ingest policy/truth |
| `/atlan/resonate` | POST | Validate text |
| `/analytics/status` | GET | System status |
| `/analytics/heatmap` | GET | Dissonance hotspots |

## Response Status Values

| Status | Meaning | Action |
|--------|---------|--------|
| `RESONANT` | Matches policy | ALLOW |
| `DISSONANT` | Violates policy | BLOCK |
| `AMBIGUOUS` | Uncertain | Review |

## Dissonance Reasons

| Reason | Example |
|--------|---------|
| `Foreign Concepts: [words]` | Unknown vocabulary |
| `Unknown Numbers: [nums]` | Numbers not in policy |
| `Negation Mismatch` | "is not" when "is" expected |
| `Number Mismatch: [x] vs [y]` | Wrong number |
| `No matching policy found` | No similar truth |

## Performance

- **Latency**: 13ms average, 21ms P95
- **Throughput**: 75+ requests/second per instance
- **Memory**: ~100MB base + policies
- **Deterministic**: Same input = same output, always

## Next Steps

1. Read [API_REFERENCE.md](./API_REFERENCE.md) for full endpoint documentation
2. Read [BENCHMARK_REPORT.md](./BENCHMARK_REPORT.md) for performance analysis
3. Run `python benchmark_comprehensive.py` to test all 73 test cases
