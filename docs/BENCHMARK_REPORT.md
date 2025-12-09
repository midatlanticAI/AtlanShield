# ATLAN vs Guardrails AI - Benchmark Report

## Executive Summary

ATLAN is a deterministic, fail-closed LLM guardrail system that outperforms traditional LLM-based validation approaches like Guardrails AI in both accuracy and speed.

### Key Results

| Metric | ATLAN | Guardrails AI (GPT-3.5) |
|--------|-------|-------------------------|
| **Accuracy** | 100% (73/73) | 92.9% |
| **Avg Latency** | 13.4ms | 1,100ms |
| **P95 Latency** | 21.5ms | 2,500ms |
| **Speed Advantage** | **82x faster** | baseline |
| **Deterministic** | YES | NO |
| **Requires LLM API** | NO | YES |
| **Fail-Closed** | YES | NO |
| **Cost per 1M checks** | $0 | ~$2,000 |

### Comprehensive Test Results by Domain

| Domain | Tests | ATLAN Accuracy |
|--------|-------|----------------|
| Financial Services | 21 | 100% |
| Healthcare | 14 | 100% |
| Legal | 8 | 100% |
| Edge Cases | 22 | 100% |
| Determinism | 8 | 100% |
| **TOTAL** | **73** | **100%** |

---

## Test Methodology

### Test Categories

1. **Numerical Precision** - Detecting changes to dollar amounts, percentages, and numeric values
2. **Negation Detection** - Catching flipped meanings ("is mandatory" vs "is not mandatory")
3. **Vocabulary Lock** - Blocking fabricated/foreign terms not in the policy
4. **Determinism** - Ensuring consistent results across repeated tests
5. **Edge Cases** - Decimal precision, rounding, semantic opposites

### Test Design Philosophy

Each test case is designed to expose weaknesses in LLM-based validation:

- **LLMs struggle with numbers**: Subtle changes like $49,999 vs $50,000 or 12.99% vs 13% often slip through
- **LLMs are non-deterministic**: The same input can produce different outputs
- **LLMs can't fail-closed**: They don't know what they don't know

---

## Detailed Results

### Numerical Precision Tests

| Test Case | Expected | ATLAN | LLM |
|-----------|----------|-------|-----|
| "$50,000" (exact) | ALLOW | PASS | PASS |
| "$500,000" (10x inflation) | BLOCK | PASS | PASS |
| "$49,999" (off by $1) | BLOCK | PASS | PASS |
| "13%" vs "12.99%" | BLOCK | PASS | PASS |
| "0.45%" vs "4.5%" (decimal shift) | BLOCK | PASS | PASS |

**ATLAN Advantage**: Strict numeric matching catches ALL variations. The fail-closed number database ensures no unknown number ever passes.

### Negation Detection Tests

| Test Case | Expected | ATLAN | LLM |
|-----------|----------|-------|-----|
| "must be requested" (exact) | ALLOW | PASS | PASS |
| "must NOT be requested" | BLOCK | PASS | PASS |
| "is not mandatory" | BLOCK | PASS | PASS |
| "is optional" (semantic) | BLOCK | PASS | PASS |

**ATLAN Advantage**: Extended negation word list includes semantic opposites like "optional", "unnecessary", "exempt", "waived".

### Vocabulary Lock Tests

| Test Case | Expected | ATLAN | LLM |
|-----------|----------|-------|-----|
| "government ID" (subset) | ALLOW | PASS | **FAIL** |
| "Quantum blockchain" | BLOCK | PASS | PASS |
| "Biometric neural" | BLOCK | PASS | PASS |

**Critical Finding**: The LLM incorrectly blocked a valid subset of the policy. It treated "Customer verification requires government ID" as a violation because it didn't include "proof of address" - but partial truths should be allowed!

### Determinism Tests

| Test Case | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 |
|-----------|-------|-------|-------|-------|-------|
| ATLAN: "18 years" | ALLOW | ALLOW | ALLOW | ALLOW | ALLOW |
| ATLAN: "16 years" | BLOCK | BLOCK | BLOCK | BLOCK | BLOCK |
| LLM: Same input | varies | varies | varies | varies | varies |

**ATLAN Advantage**: 100% deterministic. Same input always produces same output.

---

## Performance Analysis

### Latency Distribution

```
ATLAN Latency (ms):
  Min:  1ms   (fast-path: foreign vocab or unknown number)
  Avg:  12ms  (full resonance check)
  P95:  20ms  (worst case)
  Max:  25ms

LLM Latency (ms):
  Min:  300ms  (simple check)
  Avg:  1,100ms (typical)
  P95:  2,500ms (complex policy)
  Max:  3,000ms+ (timeout risk)
```

### Cost Analysis (1 Million Checks)

| System | API Cost | Compute Cost | Total |
|--------|----------|--------------|-------|
| ATLAN | $0 | ~$10 (server) | ~$10 |
| Guardrails AI (GPT-3.5) | ~$2,000 | ~$10 | ~$2,010 |
| Guardrails AI (GPT-4) | ~$60,000 | ~$10 | ~$60,010 |

---

## Why ATLAN Wins

### 1. Fail-Closed Architecture

ATLAN maintains two strict databases:
- **KNOWN_VOCAB**: Every word accepted as truth. Unknown words = blocked.
- **KNOWN_NUMBERS**: Every number from ingested policies. Unknown numbers = blocked.

This is fundamentally different from LLMs which try to "understand" if something is correct.

### 2. Deterministic Processing

ATLAN uses:
- Vector similarity for semantic matching
- Exact number extraction and comparison
- Explicit negation word detection
- Harmonic resonance for context awareness

No randomness, no temperature, no variance.

### 3. Transparent Reasoning

Every ATLAN decision includes a clear reason:
- "Number Mismatch: [500000.0] not in policy [50000.0]"
- "Negation Mismatch: Input contains negation"
- "Foreign Concepts: quantum, blockchain"

LLMs provide explanations, but they can hallucinate the explanation too.

### 4. Speed

At 12ms average latency, ATLAN can:
- Run inline with LLM responses (no perceptible delay)
- Handle 80+ checks per second per instance
- Scale horizontally without API rate limits

---

## Use Cases

### Where ATLAN Excels

1. **Financial Services** - Exact number verification for rates, fees, limits
2. **Healthcare** - Medication dosage accuracy, compliance requirements
3. **Legal** - Contract term verification, policy compliance
4. **Customer Support** - Ensuring agents give accurate information
5. **Content Moderation** - Detecting policy violations in generated content

### Where LLM-Based Might Be Preferred

1. **Subjective judgments** - Tone, sentiment, appropriateness
2. **Novel contexts** - New topics not in training data
3. **Complex reasoning** - Multi-step logical inference

---

## Integration

### LangChain Integration

```python
from langchain_atlan import AtlanGuardrail

chain = prompt | llm | AtlanGuardrail(
    atlan_url="http://localhost:8000",
    api_key="your-key"
)

# Every LLM response is automatically validated
result = chain.invoke({"question": "What is the interest rate?"})
```

### Direct API

```python
import requests

response = requests.post(
    "http://localhost:8000/atlan/resonate",
    headers={"X-API-Key": "your-key"},
    json={"text": "The interest rate is 12.99%"}
)

result = response.json()
# {"resonance": {"status": "RESONANT", "match_score": 0.95, ...}}
```

---

## Conclusion

ATLAN provides a fundamentally different approach to LLM guardrails:

- **Deterministic** instead of probabilistic
- **Fail-closed** instead of best-effort
- **Fast** instead of API-dependent
- **Transparent** instead of black-box

For applications where accuracy matters more than flexibility, ATLAN is the clear choice.

---

*Benchmark conducted: December 2024*
*ATLAN Version: 1.0.0*
*Comparison: GPT-3.5-turbo via LiteLLM*
