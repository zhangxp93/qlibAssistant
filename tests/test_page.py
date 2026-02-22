import pytest
import os
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
from pathlib import Path
import os
import sys

root_dir = Path(__file__).resolve().parent.parent
roll_dir = os.path.join(root_dir, "script")
if roll_dir not in sys.path:
    sys.path.insert(0, roll_dir)
import gen_page

# 1. 模拟环境配置：使用临时目录代替生产目录
@pytest.fixture(autouse=True)
def mock_settings(tmp_path):
    """
    自动将 gen_page 中的路径重定向到 Pytest 提供的临时目录，
    防止测试污染项目根目录。
    """
    test_data_dir = tmp_path / "source_data"
    test_build_dir = tmp_path / "site_project"
    test_data_dir.mkdir()

    # 模拟一个源 CSV 文件
    (test_data_dir / "test_score.csv").write_text("id,score\n1,0.95", encoding="utf-8")

    with patch("gen_page.DATA_DIR", str(test_data_dir)), \
         patch("gen_page.BUILD_DIR", str(test_build_dir)):
        yield test_data_dir, test_build_dir

# 2. 测试项目初始化逻辑
def test_setup_project(mock_settings):
    _, build_dir = mock_settings
    gen_page.setup_project()

    # 验证目录是否创建
    assert (Path(build_dir) / "docs" / "stylesheets").exists()
    # 验证 mkdocs.yml 是否生成且包含关键配置
    yml_content = (Path(build_dir) / "mkdocs.yml").read_text()
    assert "use_directory_urls: false" in yml_content
    assert "table-reader" in yml_content

# 3. 测试 Markdown 页面生成逻辑
def test_scan_and_generate(mock_settings):
    _, build_dir = mock_settings
    gen_page.setup_project()
    gen_page.scan_and_generate()

    docs_dir = Path(build_dir) / "docs"
    # 验证首页是否生成
    assert (docs_dir / "index.md").exists()

    # 验证 CSV 对应的 Markdown 页面及其内容
    md_file = docs_dir / "test_score.csv.md"
    assert md_file.exists()
    content = md_file.read_text()
    assert "read_csv" in content
    assert "test_score.csv" in content

# 4. 测试构建命令调用逻辑
@patch("gen_page.subprocess.run")
def test_build_site_call(mock_run, mock_settings):
    _, build_dir = mock_settings
    # 模拟子进程成功执行
    mock_run.return_value = MagicMock(returncode=0)

    gen_page.build_site()

    # 验证是否在正确的目录下执行了 mkdocs build 命令
    mock_run.assert_called_once()
    args, kwargs = mock_run.call_args
    assert "mkdocs" in args[0]
    assert "build" in args[0]
    assert kwargs["cwd"] == str(build_dir)

# 5. 测试 CSS 生成逻辑
def test_create_custom_css(mock_settings):
    _, build_dir = mock_settings
    gen_page.setup_project()
    gen_page.create_custom_css()

    css_path = Path(build_dir) / "docs" / "stylesheets" / "extra.css"
    assert css_path.exists()
    css_content = css_path.read_text()
    # 验证是否包含你写的宽度限制逻辑
    assert "max-width: 98% !important" in css_content
