import os
import re
import sys
import subprocess
import hashlib
import datetime
from typing import Optional, Tuple, List, Union
from pathlib import Path

# 第三方库
import requests
from bs4 import BeautifulSoup
import akshare as ak
import pandas as pd
from loguru import logger

# --- 常量定义 ---
DEFAULT_ENCODING = "utf-8"
DEFAULT_TIMEOUT = (10, 30)  # (连接超时, 读取超时)
GITHUB_ASSETS_PATTERN = re.compile(r"expanded_assets")
# MLflow 路径修复正则：使用命名组提高可读性
MLFLOW_PATH_PATTERN = re.compile(
    r"^(?P<key>artifact_(location|uri)):\s+file:///home/[^/]+/(?P<suffix>.*)"
)

def check_match(str_a: str, pattern: str) -> bool:
    """检查字符串是否匹配正则"""
    return bool(re.search(pattern, str_a))

def filter_csv(file_path: Union[str, Path]) -> str:
    """过滤 CSV 并返回股票代码字符串"""
    try:
        df = pd.read_csv(file_path)
        # 链式过滤提高性能
        mask = (df['avg_score'] > 0) & (df['pos_ratio'] > 0.8)
        filtered_df = df[mask].copy()

        def clean_code(text):
            return "".join(re.findall(r'\d+', str(text)))

        stock_codes = filtered_df['instrument'].apply(clean_code).tolist()
        return ",".join(stock_codes)
    except Exception as e:
        logger.error(f"过滤 CSV 失败: {e}")
        return ""

def check_match_in_list(target_str: str, regex_list: List[str]) -> bool:
    """检查目标字符串是否匹配列表中的任一正则"""
    return any(re.search(pattern, target_str) for pattern in regex_list)

def get_latest_url(base_url: str) -> str:
    """追踪 URL 重定向获取最终地址"""
    try:
        response = requests.get(base_url, allow_redirects=True, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        logger.info(f"Final download URL: {response.url}")
        return response.url
    except requests.RequestException as e:
        logger.error(f"追踪 URL 失败: {e}")
        return base_url

def run_command(cmd: str) -> Tuple[int, str, str]:
    """执行 Shell 命令并捕获输出"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            encoding=DEFAULT_ENCODING
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        return -1, "", str(e)

def calculate_file_sha256(file_path: Union[str, Path]) -> Optional[str]:
    """计算文件的 SHA256 哈希值"""
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except FileNotFoundError:
        logger.warning(f"文件未找到: {file_path}")
        return None

def get_real_github_hash(repo_url: str, target_filename: str) -> Optional[str]:
    """从 GitHub Release 页面提取资源的 SHA256"""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    try:
        session = requests.Session()
        response = session.get(repo_url, headers=headers, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        fragment_tag = soup.find("include-fragment", src=GITHUB_ASSETS_PATTERN)

        target_soup = soup
        if fragment_tag:
            assets_url = fragment_tag['src']
            if not assets_url.startswith("http"):
                assets_url = f"https://github.com{assets_url}"
            resp_assets = session.get(assets_url, headers=headers, timeout=DEFAULT_TIMEOUT)
            target_soup = BeautifulSoup(resp_assets.text, 'html.parser')

        # 定位文件链接
        file_link = target_soup.find("a", href=re.compile(re.escape(target_filename) + r"$"))
        if not file_link:
            logger.error(f"GitHub 上未找到文件: {target_filename}")
            return None

        # 向上搜索哈希剪贴板
        current = file_link.parent
        for _ in range(5):
            if not current: break
            clipboard = current.find("clipboard-copy")
            if clipboard and clipboard.get("value") and "sha256" in clipboard["value"]:
                return clipboard["value"][7:] # 移除 'sha256 ' 前缀
            current = current.parent

    except Exception as e:
        logger.error(f"解析 GitHub 哈希失败: {e}")
    return None

def append_to_file(file_path: Union[str, Path], content: str):
    """追加内容到文件"""
    try:
        with open(file_path, mode='a', encoding=DEFAULT_ENCODING) as f:
            f.write(content)
    except Exception as e:
        logger.error(f"文件追加失败: {e}")

def process_stock_code_v2(code: str) -> str:
    """标准化股票代码（带市场前缀）"""
    code_str = str(code).zfill(6)
    if code_str.startswith("6"):
        return f"SH{code_str}"
    elif code_str.startswith(("0", "3")):
        return f"SZ{code_str}"
    elif code_str.startswith(("8", "4", "920")):
        return f"BJ{code_str}"
    return f"unknown{code_str}"

def get_normalized_stock_list() -> Optional[pd.DataFrame]:
    """获取并标准化 AkShare 股票列表"""
    # 清理代理环境
    for env_key in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
        os.environ.pop(env_key, None)

    try:
        df = ak.stock_info_a_code_name()
        df['code'] = df['code'].apply(process_stock_code_v2)
        return df
    except Exception as e:
        logger.error(f"AkShare 数据拉取失败: {e}")
        return None

def get_latest_trade_date_ak():
    """获取最近一个已收盘的交易日"""
    try:
        trade_calendar = ak.tool_trade_date_hist_sina()
        today = datetime.datetime.now().date()
        past_days = [d for d in trade_calendar['trade_date'].tolist() if d <= today]

        if not past_days: return None

        last_day = past_days[-1]
        # 如果是今天且未收盘（15:00），取前一天
        if today == last_day and datetime.datetime.now().hour < 15:
            return past_days[-2] if len(past_days) > 1 else last_day
        return last_day
    except Exception as e:
        logger.error(f"获取交易日历失败: {e}")
        return None

def get_local_data_date(provider_uri):
    _, stdout, _ = run_command(f"tail -n 1 {provider_uri}/calendars/day.txt")
    return stdout

def fix_mlflow_paths(mlruns_dir: Optional[str] = None):
    """精准修复 MLflow 配置文件中的用户路径"""
    current_home = str(Path.home())
    current_prefix = f"file://{current_home}"

    if mlruns_dir is None:
        mlruns_dir = os.path.join(current_home, ".qlibAssistant", "mlruns")

    base_path = Path(mlruns_dir).expanduser().resolve()
    if not base_path.exists():
        logger.warning(f"目录不存在: {base_path}")
        return

    logger.info(f"正在修复 MLflow 路径前缀为: {current_home}")

    fix_count = 0
    for root, _, files in os.walk(base_path):
        if "meta.yaml" in files:
            yaml_path = Path(root) / "meta.yaml"
            try:
                content = yaml_path.read_text(encoding=DEFAULT_ENCODING)
                # 使用命名分组替换
                new_content, count = MLFLOW_PATH_PATTERN.subn(
                    rf"\g<key>: {current_prefix}/\g<suffix>", content
                )
                if count > 0:
                    yaml_path.write_text(new_content, encoding=DEFAULT_ENCODING)
                    fix_count += count
            except Exception as e:
                logger.error(f"修复失败 {yaml_path}: {e}")

    logger.info(f"路径修复完成，共修改 {fix_count} 处。")
