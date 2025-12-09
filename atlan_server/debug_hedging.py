import sys
import os
sys.path.append(os.path.abspath(".."))

from main import has_hedging, HEDGING_WORDS

text = "possibly digital products are non-refundable"
policy = "Digital products are non-refundable"

print(f"Text: '{text}'")
print(f"Has Hedging: {has_hedging(text)}")

print(f"Policy: '{policy}'")
print(f"Has Hedging: {has_hedging(policy)}")

print(f"'possibly' in HEDGING_WORDS: {'possibly' in HEDGING_WORDS}")
