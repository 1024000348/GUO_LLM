# check_and_fix_deps.py
import subprocess
import sys

# 要检查和自动安装的依赖列表（根据你目前 Chroma + LangChain 项目整理）
REQUIRED_PACKAGES = [
    "chromadb",          # 向量数据库
    "langchain",         # LangChain 主包
    "langchain_community",  # LangChain 社区扩展
    "zhipuai",           # 智谱 API
    "python-dotenv",     # 读取 .env
    "PyMuPDF",           # 处理 PDF
    "unstructured",      # 处理 markdown
    "posthog==3.0.1",    # 避免新版本兼容问题
    "monotonic"          # posthog 依赖
]

def install_package(package):
    """安装指定的 Python 包"""
    print(f"🔍 检测到缺失依赖：{package}，正在安装...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])

def check_and_install():
    """检查并安装缺失依赖"""
    import pkg_resources
    installed_packages = {pkg.key for pkg in pkg_resources.working_set}

    for pkg in REQUIRED_PACKAGES:
        pkg_name = pkg.split("==")[0]  # 去掉版本号
        if pkg_name.lower() not in installed_packages:
            install_package(pkg)
        else:
            print(f"✅ 已安装：{pkg}")

if __name__ == "__main__":
    print("📦 正在检查依赖环境...")
    check_and_install()
    print("✨ 依赖检查完成，可以安全运行项目！")
