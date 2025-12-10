import re
from typing import List, Tuple, Optional, Any

# ==========================================
# 1. Extraction Utilities (Refactored from main.py)
# ==========================================

def evaluate_math_expression(text: str) -> List[float]:
    """Extracts and evaluates simple math expressions (e.g., '10 + 5')."""
    math_pattern = r'(\d+\.?\d*)[\s\$]*(plus|\+|minus|\-|times|\*|divided by|\/)[\s\$]*(\d+\.?\d*)'
    op_map = {'plus': '+', 'minus': '-', 'times': '*', 'divided by': '/'}
    results = []

    for n1, op_str, n2 in re.findall(math_pattern, text, re.IGNORECASE):
        try:
            val1, val2 = float(n1), float(n2)
            op = op_map.get(op_str.lower(), op_str)
            if op == '+': res = val1 + val2
            elif op == '-': res = val1 - val2
            elif op == '*': res = val1 * val2
            elif op == '/': res = val1 / val2 if val2 != 0 else 0
            else: continue
            results.append(res)
        except: pass
    return results

def extract_values_with_context(text: str) -> List[dict]:
    """
    Extracts values along with their units and surrounding context keywords.
    Returns: [{'value': 50.0, 'unit': 'usd', 'raw': '$50', 'type': 'currency'}]
    """
    normalized = re.sub(r'(\d),(\d{3})', r'\1\2', text.lower()) # Normalize commas
    extracted = []

    # Currency
    currency_pattern = r'(?:\$|usd\s?)(\d+(?:\.\d+)?)'
    for val in re.findall(currency_pattern, normalized):
        extracted.append({'value': float(val), 'unit': 'usd', 'type': 'currency'})

    # Time / Duration
    time_pattern = r'(\d+(?:\.\d+)?)\s*(days?|hours?|minutes?|weeks?|months?|years?)'
    for val, unit in re.findall(time_pattern, normalized):
        # Normalize time to "days" for comparison? For now just keep unit.
        extracted.append({'value': float(val), 'unit': unit.rstrip('s'), 'type': 'temporal'})

    # Percentages
    pct_pattern = r'(\d+(?:\.\d+)?)\s*(?:%|percent)'
    for val in re.findall(pct_pattern, normalized):
         extracted.append({'value': float(val), 'unit': '%', 'type': 'percentage'})

    return extracted

# ==========================================
# 2. Policy Parsing (Constraint Detection)
# ==========================================

def parse_policy_constraint(policy_text: str) -> List[dict]:
    """
    Parses a policy text to find constraints.
    Example: "Refunds limited to $50" -> {'operator': '<=', 'limit': 50.0, 'unit': 'usd'}
    Example: "Must be over 18 years" -> {'operator': '>=', 'limit': 18.0, 'unit': 'year'}
    """
    constraints = []
    text_lower = policy_text.lower()

    # 1. Identify Limit Keywords
    # "Up to", "Max", "Limit", "Maximum", "Not exceed", "Under", "Less than"
    max_keywords = ["max", "maximum", "up to", "limit", "not exceed", "under", "less than", "below"]
    # "Min", "Minimum", "At least", "Over", "More than", "Great than"
    min_keywords = ["min", "minimum", "at least", "over", "more than", "greater than", "above"]

    values = extract_values_with_context(text_lower)

    for val in values:
        # Check context around the value
        # This is a simple proximity check (naive but fast)
        # In a real NLP system we'd use Dependency Parsing.

        # Default assumption: Policy stated numbers are often LIMITS (Upper bounds).
        # "Refund limit is $50" -> <= 50.
        operator = None

        # Check explicit operators in text
        for kw in max_keywords:
            if kw in text_lower: # Naive: Global check in sentence
                 operator = "<="
                 break

        if not operator:
            for kw in min_keywords:
                if kw in text_lower:
                    operator = ">="
                    break

        # If no keyword, but it's a policy doc...
        # "Refunds: $50" usually implies Max.
        # But we should be careful.
        # Default: If we find a number in a "Constraint" node, treat it as an Upper Bound?
        # Let's stick to explicit keywords for V1 to reduce False Positives.

        if operator:
             constraints.append({
                 'operator': operator,
                 'limit': val['value'],
                 'unit': val['unit'],
                 'type': val['type']
             })

    return constraints

# ==========================================
# 3. Compliance Validation
# ==========================================

def validate_compliance(user_text: str, policy_nodes: List[str]) -> Tuple[bool, List[str]]:
    """
    Compares user output against policy constraints.
    Returns: (is_compliant, violations)

    Enhanced: Also checks COMPUTED TOTALS (e.g., "$960 plus $906" = $1866 vs $1000 limit)
    """
    violations = []
    is_compliant = True

    # 1. Extract values proposed by User (LLM)
    user_values = extract_values_with_context(user_text)

    # 2. Check for math operations that compute a total (spending checks)
    computed_totals = evaluate_math_expression(user_text)

    if not user_values and not computed_totals:
        return True, [] # No numbers, nothing to check (Pass Semantic layer separately)

    for policy_text in policy_nodes:
        constraints = parse_policy_constraint(policy_text)

        for constraint in constraints:
            c_type = constraint['type']
            c_unit = constraint['unit']
            c_limit = constraint['limit']
            op = constraint['operator']

            # Check individual values
            for u_val in user_values:
                # Type/Unit Mismatch Check
                if u_val['type'] != c_type:
                    continue
                if u_val['unit'] != c_unit:
                    continue

                val = u_val['value']

                # Check Logic
                violation = False
                if op == "<=" and val > c_limit:
                    violation = True
                elif op == ">=" and val < c_limit:
                    violation = True
                elif op == "<" and val >= c_limit:
                    violation = True
                elif op == ">" and val <= c_limit:
                    violation = True

                if violation:
                    is_compliant = False
                    violations.append(f"Value '{val} {c_unit}' violates constraint '{op} {c_limit} {c_unit}' in policy.")

            # Check computed totals (for spending statements like "$X plus $Y")
            # Only check against currency limits
            if c_type == 'currency' and computed_totals:
                for total in computed_totals:
                    violation = False
                    if op == "<=" and total > c_limit:
                        violation = True
                    elif op == ">=" and total < c_limit:
                        violation = True

                    if violation:
                        is_compliant = False
                        violations.append(f"Computed total '{total} {c_unit}' violates constraint '{op} {c_limit} {c_unit}' in policy.")

    return is_compliant, violations
