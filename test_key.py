# test_key.py
import os, sys
from dotenv import load_dotenv, find_dotenv

# 尝试加载 .env（如果你使用 .env 管理）
load_dotenv(find_dotenv())

print("Python exe:", sys.executable)
print("ZHIPUAI_API_KEY:", os.environ.get("ZHIPUAI_API_KEY"))
print("ZHIPU_API_KEY:", os.environ.get("ZHIPU_API_KEY"))

try:
    import zhipuai
    print("zhipuai package path:", getattr(zhipuai, "__file__", "built-in/module"))
except Exception as e:
    print("import zhipuai error:", e)
  