import sys
try:
    import atlan
    from atlan.memory import RCoreMemoryIndex
    print(f"Successfully imported atlan package from: {atlan.__file__}")
except ImportError as e:
    print(f"Failed to import atlan: {e}")
    sys.exit(1)

def test_package():
    print("Testing Atlan Core Package...")
    memory = RCoreMemoryIndex()
    idx = memory.add("Hello World", sentiment_score=1.0)
    print(f"Added node with ID: {idx}")
    
    results = memory.search(memory.memory[idx].vector)
    print(f"Search Result: {memory.memory[results[0][0]].phrase}")
    
    if results[0][0] == idx:
        print("PASS: Package is functional.")
    else:
        print("FAIL: Search failed.")

if __name__ == "__main__":
    test_package()
