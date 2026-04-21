#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地一键调度脚本：自动推送模型到 GitHub Actions 转换，完成后下载结果。

使用前准备：
  1. 在 GitHub 上创建一个空仓库（不要勾选 README / .gitignore）
  2. 把仓库 HTTPS 地址粘贴到下方 GITHUB_REPO_URL
  3. 确保已安装 GitHub CLI 并已登录：运行 `gh auth login` 按提示操作
  4. 把 SOURCE_PT 改成你训练好的 best.pt 绝对路径
  5. 把 SOURCE_CALIB 改成你的校准图目录

以后每次转换，只需运行：
    python github_converter.py
"""

import os
import sys
import random
import shutil
import subprocess
import time
import json
from pathlib import Path

# ========================== 首次配置（只需改这里） ==========================
# GitHub 仓库地址（例如 https://github.com/你的用户名/k230-converter.git）
GITHUB_REPO_URL = "https://github.com/X7997/k230-converter.git"

# 训练好的模型路径（直接把绝对地址粘贴过来）
SOURCE_PT = r"Q:\K230_ultralytics\ultralytics_main\Fridgify.v1-fridge_annotated_5132.yolov12\runs\detect\results\fridge_yolo12n\weights\best.pt"

# 校准图目录（用于 PTQ 量化）
SOURCE_CALIB = r"Q:\K230_ultralytics\ultralytics_main\Fridgify.v1-fridge_annotated_5132.yolov12\valid\images"

# 每次推送多少张校准图到云端（太多会慢，建议 20~50）
MAX_CALIB_UPLOAD = 30
# ==========================================================================

REPO_ROOT = Path(__file__).parent.resolve()
MODELS_DIR = REPO_ROOT / "models"
CALIB_DIR = REPO_ROOT / "calib"
OUTPUT_DIR = REPO_ROOT / "output"


def run_cmd(cmd: str, check: bool = True, capture: bool = True):
    """执行 shell 命令并打印输出"""
    print(f">>> {cmd}")
    kwargs = {"shell": True, "text": True}
    if capture:
        kwargs["capture_output"] = True
    result = subprocess.run(cmd, **kwargs)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if check and result.returncode != 0:
        raise RuntimeError(f"命令失败: {cmd}")
    return result


def check_gh_login():
    """检查 gh CLI 是否已登录"""
    res = run_cmd("gh auth status", check=False)
    if res.returncode != 0:
        print("\n❌ GitHub CLI (gh) 未登录。请先在终端运行以下命令并跟随提示操作：")
        print("   gh auth login")
        print("   （选择 HTTPS 或 SSH，按提示浏览器授权即可）\n")
        sys.exit(1)
    print("✅ GitHub CLI 已登录")


def init_git():
    """初始化本地 git 仓库并关联远程"""
    git_dir = REPO_ROOT / ".git"
    if not git_dir.exists():
        run_cmd("git init")
        print("✅ 已初始化本地 git 仓库")

    res = run_cmd("git remote get-url origin", check=False)
    if res.returncode != 0:
        if not GITHUB_REPO_URL:
            print("❌ 错误：请先在脚本里填写 GITHUB_REPO_URL")
            print("   步骤：在 GitHub 网页新建空仓库 -> 复制 HTTPS 地址 -> 粘贴到脚本里")
            sys.exit(1)
        run_cmd(f"git remote add origin {GITHUB_REPO_URL}")
        print(f"✅ 已关联远程仓库: {GITHUB_REPO_URL}")
    else:
        print(f"✅ 远程仓库已关联: {res.stdout.strip()}")


def prepare_files():
    """把模型和校准图复制到仓库目录，准备推送"""
    MODELS_DIR.mkdir(exist_ok=True)
    CALIB_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 复制模型
    if not os.path.isfile(SOURCE_PT):
        print(f"❌ 找不到模型文件: {SOURCE_PT}")
        sys.exit(1)
    dest_pt = MODELS_DIR / "input.pt"
    shutil.copy2(SOURCE_PT, dest_pt)
    print(f"✅ 已复制模型 -> {dest_pt}  ({dest_pt.stat().st_size / 1024 / 1024:.1f} MB)")

    # 复制校准图
    if not os.path.isdir(SOURCE_CALIB):
        print(f"❌ 找不到校准图目录: {SOURCE_CALIB}")
        sys.exit(1)

    images = [p for p in Path(SOURCE_CALIB).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
    if not images:
        print("❌ 校准图目录里没有支持的图片格式（jpg/jpeg/png/bmp）")
        sys.exit(1)

    # 清空旧校准图
    for old in CALIB_DIR.iterdir():
        old.unlink()

    if len(images) <= MAX_CALIB_UPLOAD:
        selected = images
    else:
        random.seed(42)
        selected = random.sample(images, MAX_CALIB_UPLOAD)

    for img in selected:
        shutil.copy2(img, CALIB_DIR / img.name)

    print(f"✅ 已复制 {len(selected)} 张校准图 -> {CALIB_DIR}")


def push_to_github():
    """提交并强制推送到 convert-request 分支"""
    run_cmd("git checkout -B convert-request")
    # 用 -f 强制添加被 .gitignore 拦截的文件（.pt / .jpg 等）
    run_cmd("git add -f models/ calib/")
    run_cmd("git add .github/ convert_k230.py github_converter.py .gitignore")
    # 显示将要提交的文件，帮助调试
    run_cmd("git status", check=False)
    res = run_cmd('git commit -m "Request K230 conversion"', check=False)
    if res.returncode != 0:
        print("⚠️ 没有新的变更需要提交，仍将推送以触发 Actions")
    run_cmd("git push -u origin convert-request --force")
    print("✅ 已推送到 GitHub（convert-request 分支）")


def wait_and_download():
    """等待 GitHub Actions 完成并自动下载 artifact"""
    print("\n⏳ 等待 GitHub Actions 启动（最多 30 秒）...")
    time.sleep(8)

    res = run_cmd('gh run list --branch convert-request --limit 1 --json databaseId,status,conclusion,name', check=False)
    if res.returncode != 0 or not res.stdout.strip():
        print("⚠️ 无法获取 Actions 运行列表，请手动去仓库 Actions 页面查看")
        return False

    try:
        runs = json.loads(res.stdout.strip())
    except json.JSONDecodeError:
        print("⚠️ 解析 Actions 列表失败")
        return False

    if not runs:
        print("⚠️ 未找到运行记录，可能推送尚未触发 Actions")
        return False

    run_id = runs[0]["databaseId"]
    print(f"🔍 检测到运行 ID: {run_id} ({runs[0].get('name', '')})")
    print("⏳ 开始自动等待转换完成（按 Ctrl+C 可中断）...\n")

    watch_res = run_cmd(f"gh run watch {run_id} --exit-status", check=False, capture=False)
    if watch_res.returncode != 0:
        print("\n❌ GitHub Actions 运行失败或中断")
        # 尝试打印失败日志帮助调试
        log_res = run_cmd(f"gh run view {run_id} --log-failed", check=False)
        if log_res.returncode == 0 and log_res.stdout:
            print("--- 失败日志（最后 3000 字符）---")
            print(log_res.stdout[-3000:] if len(log_res.stdout) > 3000 else log_res.stdout)
            print("--- 日志结束 ---")
        return False

    print("\n✅ Actions 运行成功！开始下载结果...")

    # 清理旧输出
    for f in OUTPUT_DIR.iterdir():
        if f.is_file():
            f.unlink()
        elif f.is_dir():
            shutil.rmtree(f)

    dl_res = run_cmd(f'gh run download {run_id} --name k230-model --dir "{OUTPUT_DIR}"', check=False)
    if dl_res.returncode != 0:
        print("⚠️ 下载指定 artifact 失败，尝试下载全部 artifact...")
        dl_res = run_cmd(f'gh run download {run_id} --dir "{OUTPUT_DIR}"', check=False)
        if dl_res.returncode != 0:
            return False

    # 查找下载到的 kmodel
    kmodels = list(OUTPUT_DIR.rglob("*.kmodel"))
    if kmodels:
        print("\n🎉 转换成功！模型已下载到本地:")
        for k in kmodels:
            size_kb = k.stat().st_size / 1024
            print(f"   📦 {k}  ({size_kb:.1f} KB)")
        for k in kmodels:
            final = REPO_ROOT / k.name
            shutil.copy2(k, final)
            print(f"   📋 已复制到根目录: {final}")
    else:
        print("⚠️ 未在下载结果中找到 .kmodel 文件")
        print(f"   请检查 {OUTPUT_DIR} 目录下的内容")
    return True


def main():
    print("=" * 60)
    print(" GitHub Actions K230 模型转换调度器")
    print("=" * 60)

    check_gh_login()
    init_git()
    prepare_files()
    push_to_github()
    success = wait_and_download()

    if not success:
        url = GITHUB_REPO_URL.rstrip(".git") + "/actions" if GITHUB_REPO_URL else ""
        print(f"\n💡 请手动打开 GitHub Actions 页面查看或下载结果:")
        if url:
            print(f"   {url}")
        print("   也可以运行: gh run list --branch convert-request")

    print("\n" + "=" * 60)
    print(" 流程结束")
    print("=" * 60)


if __name__ == "__main__":
    main()