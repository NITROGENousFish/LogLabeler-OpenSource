import sys, os

print(
    f"sys.path append baseline logdeep in {os.path.dirname(os.path.abspath(__file__))}"
)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))