#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Yolo_to_K230 - 一键将 YOLO 模型转换为 K230 Kmodel

用法:
    python easy_k230.py                    # 交互式引导（每次询问模型路径）
    python easy_k230.py --model=best.pt    # 命令行指定模型（跳过交互）
    python easy_k230.py --model=best.pt --calib=images/  # 完整命令行

提示:
    - 交互模式下直接回车可使用上次保存的配置
    - 配置自动保存到 config.json
"""

import os
import sys
import shutil
import subprocess
import time
import json
from pathlib import Path
from datetime import datetime

# ╔══════════════════════════════════════════════════════════════════════╗
# ║                     ★ 配置区（交互式填入后自动保存） ★                ║
# ║  首次运行会交互式引导填写，之后自动保存到 config.json 无需再改       ║
# ╚══════════════════════════════════════════════════════════════════════╝

DEFAULT_CONFIG = {
    "github_repo_url": "",
    "source_pt": "",
    "source_calib": "",
    "input_shape": [1, 3, 320, 320],
    "onnx_imgsz": 320,
    "onnx_opset": 11,
    "onnx_simplify": True,
    "onnx_nms": False,
    "quant_type": "uint8",
    "w_quant_type": "uint8",
    "calib_method": "Kld",
    "input_type": "uint8",
    "output_type": "float16",
    "target": "k230",
    "kmodel_filename": "",
    "proxy_port": 0,
    "_configured": False,
}

CONFIG_PATH = Path(__file__).parent / "config.json"

REPO_ROOT = Path(__file__).parent.resolve()
MODELS_DIR = REPO_ROOT / "models"
CALIB_DIR = REPO_ROOT / "calib"
OUTPUT_DIR = REPO_ROOT / "output"

STEPS = [
    "① 环境检查",
    "② 文件准备",
    "③ 参数同步",
    "④ 推送上传",
    "⑤ 云端转换",
    "⑥ 下载结果",
]


# ─── 进度条 ────────────────────────────────────────────────────────────

class ProgressBar:
    def __init__(self, total_steps=6):
        self.total = total_steps
        self.current = 0
        self.step_names = STEPS[:total_steps]

    def advance(self, msg=""):
        self.current += 1
        pct = self.current * 100 // self.total
        filled = "█" * self.current
        empty = "░" * (self.total - self.current)
        step_label = self.step_names[self.current - 1] if self.current <= len(self.step_names) else ""
        suffix = f" {msg}" if msg else ""
        print(f"\r  [{filled}{empty}] {pct:3d}%  {step_label}{suffix}")

    def info(self, msg):
        print(f"       {msg}")


PROGRESS = ProgressBar(6)


# ─── 配置加载/保存 ────────────────────────────────────────────────────

def load_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            saved = json.load(f)
        cfg = dict(DEFAULT_CONFIG)
        cfg.update(saved)
        return cfg
    return dict(DEFAULT_CONFIG)


def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    PROGRESS.info(f"配置已保存到 {CONFIG_PATH}")


def interactive_setup(cfg, force_all=False):
    """交互式配置，每次询问模型路径，回车使用上次配置"""
    print("\n" + "=" * 60)
    print("  Yolo_to_K230 配置")
    print("=" * 60)
    print("  提示: 直接回车使用上次保存的配置\n")

    # 模型路径 - 每次都询问
    print("📋 模型文件路径（.pt 或 .onnx）")
    if cfg.get("source_pt"):
        print(f"   上次: {cfg['source_pt']}")
    new_path = input("   新路径 [回车保持]: ").strip().strip("'\"")
    if new_path:
        cfg["source_pt"] = new_path

    if not cfg.get("source_pt"):
        print("   ❌ 模型路径不能为空")
        sys.exit(1)

    # 校准图目录 - 每次都询问
    print("\n📋 校准图目录")
    if cfg.get("source_calib"):
        print(f"   上次: {cfg['source_calib']}")
    new_calib = input("   新目录 [回车保持]: ").strip().strip("'\"")
    if new_calib:
        cfg["source_calib"] = new_calib

    if not cfg.get("source_calib"):
        print("   ❌ 校准图目录不能为空")
        sys.exit(1)

    # 首次配置或强制配置时，询问仓库地址和高级参数
    if force_all or not cfg.get("github_repo_url"):
        print("\n📋 GitHub 仓库地址")
        print("   示例: https://github.com/用户名/仓库名.git")
        if cfg.get("github_repo_url"):
            print(f"   上次: {cfg['github_repo_url']}")
        new_repo = input("   新地址 [回车保持]: ").strip().strip("'\"")
        if new_repo:
            cfg["github_repo_url"] = new_repo

    if not cfg.get("github_repo_url"):
        print("   ❌ 仓库地址不能为空")
        sys.exit(1)

    # 高级参数 - 仅首次或明确要求时询问
    if force_all or not cfg.get("_configured"):
        print("\n📋 高级参数（直接回车使用默认值）")
        print(f"   输入尺寸 [N,C,H,W]（默认 {cfg['input_shape']}）: ", end="")
        shape_str = input().strip()
        if shape_str:
            try:
                cfg["input_shape"] = [int(x.strip()) for x in shape_str.replace("[", "").replace("]", "").split(",")]
            except ValueError:
                print("   ⚠️ 格式不对，使用默认值")

        print(f"   ONNX imgsz（默认 {cfg['onnx_imgsz']}）: ", end="")
        imgsz_str = input().strip()
        if imgsz_str:
            cfg["onnx_imgsz"] = int(imgsz_str)

        if not cfg.get("kmodel_filename"):
            model_name = Path(cfg["source_pt"]).stem
            cfg["kmodel_filename"] = f"{model_name}.kmodel"
        print(f"   输出文件名（默认 {cfg['kmodel_filename']}）: ", end="")
        kfn = input().strip()
        if kfn:
            cfg["kmodel_filename"] = kfn if kfn.endswith(".kmodel") else kfn + ".kmodel"

        proxy = input("   本地代理端口（v2rayN=10808, Clash=7890, 无=0, 默认0）: ").strip()
        cfg["proxy_port"] = int(proxy) if proxy else 0

        cfg["_configured"] = True

    save_config(cfg)
    return cfg


# ─── 工具函数 ──────────────────────────────────────────────────────────

def run_cmd(cmd, check=True, capture=True, cwd=None, quiet=False):
    if not quiet:
        print(f"  >>> {cmd}")
    kwargs = {"shell": True, "encoding": "utf-8", "errors": "replace"}
    if cwd:
        kwargs["cwd"] = str(cwd)
    else:
        kwargs["cwd"] = str(REPO_ROOT)
    if capture:
        kwargs["capture_output"] = True
    result = subprocess.run(cmd, **kwargs)
    if not quiet:
        if result.stdout and len(result.stdout) < 500:
            print(result.stdout)
        if result.stderr and len(result.stderr) < 500:
            print(result.stderr, file=sys.stderr)
    if check and result.returncode != 0:
        raise RuntimeError(f"命令失败: {cmd}")
    return result


def setup_proxy(proxy_port):
    if not proxy_port:
        return
    proxy_url = f"http://127.0.0.1:{proxy_port}"
    os.environ["HTTP_PROXY"] = proxy_url
    os.environ["HTTPS_PROXY"] = proxy_url
    os.environ["ALL_PROXY"] = proxy_url
    run_cmd(f'git config --global http.proxy "{proxy_url}"', check=False, quiet=True)
    run_cmd(f'git config --global https.proxy "{proxy_url}"', check=False, quiet=True)
    print(f"  ✅ 代理已设置: {proxy_url}")


def format_duration(seconds):
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}秒"
    return f"{seconds // 60}分{seconds % 60:02d}秒"


def format_size(path):
    size = Path(path).stat().st_size
    if size > 1024 * 1024:
        return f"{size / 1024 / 1024:.1f} MB"
    return f"{size / 1024:.0f} KB"


# ─── 主要流程 ──────────────────────────────────────────────────────────

def check_gh_login():
    res = run_cmd("gh auth status", check=False, quiet=True)
    if res.returncode != 0:
        print("\n  ❌ GitHub CLI (gh) 未登录")
        print("  请先在终端运行: gh auth login")
        print("  选择 HTTPS → 浏览器授权即可\n")
        sys.exit(1)
    print("  ✅ GitHub CLI 已登录")


def init_git(repo_url):
    git_dir = REPO_ROOT / ".git"
    if not git_dir.exists():
        run_cmd("git init", quiet=True)
        print("  ✅ 已初始化本地 git 仓库")

    res = run_cmd("git remote get-url origin", check=False, quiet=True)
    current_url = res.stdout.strip() if res.returncode == 0 else ""
    if not current_url:
        run_cmd(f"git remote add origin {repo_url}", quiet=True)
        print(f"  ✅ 已关联远程仓库: {repo_url}")
    elif current_url != repo_url:
        run_cmd(f"git remote set-url origin {repo_url}", quiet=True)
        print(f"  ✅ 远程仓库已更新: {repo_url}")
    else:
        print(f"  ✅ 远程仓库已关联: {repo_url}")


def prepare_files(cfg):
    MODELS_DIR.mkdir(exist_ok=True)
    CALIB_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    source_pt = cfg["source_pt"]
    source_calib = cfg["source_calib"]

    if not os.path.isfile(source_pt):
        print(f"  ❌ 找不到模型文件: {source_pt}")
        sys.exit(1)
    dest_pt = MODELS_DIR / "input.pt"
    shutil.copy2(source_pt, dest_pt)
    print(f"  ✅ 模型  {os.path.basename(source_pt)}  ({format_size(dest_pt)})")

    if not os.path.isdir(source_calib):
        print(f"  ❌ 找不到校准图目录: {source_calib}")
        sys.exit(1)

    images = [p for p in Path(source_calib).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
    if not images:
        print("  ❌ 校准图目录里没有支持的图片格式")
        sys.exit(1)

    for old in CALIB_DIR.iterdir():
        old.unlink()

    total = len(images)
    bar_width = 30
    for i, img in enumerate(images, 1):
        shutil.copy2(img, CALIB_DIR / img.name)
        pct = i * 100 // total
        filled = "█" * (pct * bar_width // 100)
        empty = "░" * (bar_width - len(filled))
        print(f"\r  复制校准图 [{filled}{empty}] {i}/{total} ({pct}%)", end="", flush=True)
    print(f"\r  ✅ 校准图  {total} 张全部复制完成                    ")

    return total


def sync_workflow_env(cfg, calib_count):
    yml_path = REPO_ROOT / ".github" / "workflows" / "convert_k230.yml"
    if not yml_path.exists():
        print("  ⚠️ 未找到 workflow yml，跳过同步")
        return

    with open(yml_path, "r", encoding="utf-8") as f:
        content = f.read()

    ishape = cfg["input_shape"]
    env_block = (
        "          K230_MODEL_PATH: models/input.pt\n"
        f"          K230_KMODEL_PATH: results/{cfg['kmodel_filename']}\n"
        "          K230_CALIB_DIR: calib/\n"
        f'          K230_INPUT_SHAPE: "{ishape[0]},{ishape[1]},{ishape[2]},{ishape[3]}"\n'
        f'          K230_ONNX_IMGSZ: "{cfg["onnx_imgsz"]}"\n'
        f'          K230_ONNX_OPSET: "{cfg["onnx_opset"]}"\n'
        f'          K230_ONNX_SIMPLIFY: "{str(cfg["onnx_simplify"]).lower()}"\n'
        f'          K230_ONNX_NMS: "{str(cfg["onnx_nms"]).lower()}"\n'
        f'          K230_QUANT_TYPE: "{cfg["quant_type"]}"\n'
        f'          K230_W_QUANT_TYPE: "{cfg["w_quant_type"]}"\n'
        f'          K230_CALIB_METHOD: "{cfg["calib_method"]}"\n'
        f'          K230_MAX_CALIB_IMAGES: "{calib_count}"\n'
        f'          K230_TARGET: "{cfg["target"]}"\n'
        f'          K230_INPUT_TYPE: "{cfg["input_type"]}"\n'
        f'          K230_OUTPUT_TYPE: "{cfg["output_type"]}"\n'
    )

    import re
    pattern = r"          K230_MODEL_PATH:.*?\n(          K230_\w+:.*?\n)*"
    new_content = re.sub(pattern, env_block, content, count=1)

    if new_content != content:
        with open(yml_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"  ✅ 参数已同步 INPUT_SHAPE={ishape}, {calib_count}张校准图")
    else:
        print("  ℹ️ workflow 参数无变化")


def push_to_github(cfg):
    sync_workflow_env(cfg, getattr(sys.modules[__name__], '_calib_count', 999))

    run_cmd("git checkout -B convert-request", quiet=True)

    run_cmd(f'git add -f "{MODELS_DIR}/" "{CALIB_DIR}/"', quiet=True)
    run_cmd(f'git add -f "{REPO_ROOT / ".github" / "workflows" / "convert_k230.yml"}"', quiet=True)
    run_cmd(f'git add -f "{REPO_ROOT / "convert_k230.py"}"', quiet=True)
    run_cmd(f'git add -f "{REPO_ROOT / "easy_k230.py"}"', quiet=True)
    run_cmd(f'git add -f "{REPO_ROOT / "README.md"}"', quiet=True)

    gitignore_path = REPO_ROOT / ".gitignore"
    if gitignore_path.exists():
        run_cmd(f'git add -f "{gitignore_path}"', quiet=True)

    for wf in ["ci.yml", "cla.yml", "conda-check-prs.yml", "docker.yml", "docs.yml",
               "format.yml", "links.yml", "merge-main-into-prs.yml", "mirror.yml",
               "publish.yml", "stale.yml"]:
        run_cmd(f'git reset HEAD -- ".github/workflows/{wf}"', check=False, quiet=True)
    run_cmd("git reset HEAD -- .github/ISSUE_TEMPLATE/ .github/dependabot.yml", check=False, quiet=True)

    res = run_cmd('git commit -m "Request K230 conversion"', check=False, quiet=True)
    if res.returncode != 0:
        print("  ⚠️ 无新变更，仍将推送触发 Actions")
    run_cmd("git push -u origin convert-request --force", quiet=True)
    print("  ✅ 已推送到 GitHub")


def poll_run_status(run_id):
    """轮询 Actions 运行状态，显示进度条和当前步骤"""
    start = time.time()
    step_emojis = {"Setup": "⚙️", "Install": "📦", "Run conversion": "🔥", "Upload": "⬆️", "Post": "✅"}

    while True:
        res = run_cmd(
            f'gh run view {run_id} --json status,conclusion,jobs --jq '
            '"{status: .status, conclusion: .conclusion, '
            'steps: [.jobs[0].steps[] | {name: .name, status: .status, conclusion: .conclusion}]}"',
            check=False, quiet=True
        )
        if res.returncode != 0:
            time.sleep(5)
            continue

        try:
            data = json.loads(res.stdout.strip())
        except (json.JSONDecodeError, ValueError):
            time.sleep(5)
            continue

        status = data.get("status", "")
        conclusion = data.get("conclusion", "")
        steps = data.get("steps", [])

        done = sum(1 for s in steps if s.get("conclusion") == "success")
        total = max(len(steps), 1)
        pct = done * 100 // total
        filled = "█" * done
        empty = "░" * (total - done)

        current_step = ""
        for s in steps:
            if s.get("status") == "in_progress":
                current_step = s.get("name", "")
                break

        emoji = "⏳"
        for key, em in step_emojis.items():
            if key in current_step:
                emoji = em
                break

        elapsed = format_duration(time.time() - start)
        step_info = f" {emoji} {current_step}" if current_step else ""
        print(f"\r  [{filled}{empty}] {pct:3d}%{step_info}  ({elapsed})", end="", flush=True)

        if status == "completed":
            print()
            return conclusion == "success"

        time.sleep(3)


def wait_and_download(cfg):
    print("\n  ⏳ 等待 GitHub Actions 启动...")
    time.sleep(8)

    PROGRESS.advance("云端转换")
    res = run_cmd('gh run list --branch convert-request --limit 1 --json databaseId,status,conclusion,name', check=False, quiet=True)
    if res.returncode != 0 or not res.stdout.strip():
        print("  ⚠️ 无法获取 Actions 运行列表")
        return False

    try:
        runs = json.loads(res.stdout.strip())
    except json.JSONDecodeError:
        print("  ⚠️ 解析 Actions 列表失败")
        return False

    if not runs:
        print("  ⚠️ 未找到运行记录")
        return False

    run_id = runs[0]["databaseId"]
    print(f"  🔍 运行 ID: {run_id}")

    success = poll_run_status(run_id)

    if not success:
        print("  ❌ GitHub Actions 运行失败")
        log_res = run_cmd(f"gh run view {run_id} --log-failed", check=False, quiet=True)
        if log_res.returncode == 0 and log_res.stdout:
            print("  --- 失败日志（最后 20 行）---")
            lines = log_res.stdout.strip().split("\n")
            for line in lines[-20:]:
                print(f"  {line}")
        return False

    # 下载
    PROGRESS.advance("下载结果")
    for f in OUTPUT_DIR.iterdir():
        if f.is_file():
            f.unlink()
        elif f.is_dir():
            shutil.rmtree(f)

    dl_res = run_cmd(f'gh run download {run_id} --name k230-model --dir "{OUTPUT_DIR}"', check=False, quiet=True)
    if dl_res.returncode != 0:
        dl_res = run_cmd(f'gh run download {run_id} --dir "{OUTPUT_DIR}"', check=False, quiet=True)
        if dl_res.returncode != 0:
            print("  ❌ 下载失败")
            return False

    kmodels = list(OUTPUT_DIR.rglob("*.kmodel"))
    if kmodels:
        script_dir = Path(__file__).parent.resolve()
        for k in kmodels:
            final = script_dir / k.name
            shutil.copy2(k, final)
        print(f"  ✅ 已下载 {kmodels[0].name}  ({format_size(kmodels[0])})")
        print(f"  📂 保存位置: {script_dir / kmodels[0].name}")
    else:
        print("  ⚠️ 未找到 .kmodel 文件")
        return False
    return True


def main():
    total_start = time.time()

    print()
    print("  ╔══════════════════════════════════════════════════╗")
    print("  ║     🔧 Yolo_to_K230 — 一键转换 K230 Kmodel      ║")
    print("  ║     本地零环境 · GitHub Actions 云端转换          ║")
    print("  ╚══════════════════════════════════════════════════╝")
    print()

    # 命令行参数或交互式配置
    args = sys.argv[1:]
    if args:
        cfg = load_config()
        for arg in args:
            if arg.startswith("--model="):
                cfg["source_pt"] = arg.split("=", 1)[1].strip("'\"")
            elif arg.startswith("--calib="):
                cfg["source_calib"] = arg.split("=", 1)[1].strip("'\"")
            elif arg.startswith("--repo="):
                cfg["github_repo_url"] = arg.split("=", 1)[1].strip("'\"")
        save_config(cfg)
    else:
        # 每次运行都进入交互式配置
        cfg = load_config()
        cfg = interactive_setup(cfg, force_all=not cfg.get("_configured"))

    print()
    print(f"  模型:   {cfg['source_pt']}")
    print(f"  校准图: {cfg['source_calib']}")
    print(f"  尺寸:   {cfg['input_shape']}  (imgsz={cfg['onnx_imgsz']})")
    print(f"  目标:   {cfg['target']}  量化: {cfg['quant_type']}/{cfg['w_quant_type']}")
    print(f"  输入:   {cfg['input_type']}  输出: {cfg['output_type']}")
    print(f"  输出:   {cfg['kmodel_filename']}")
    print()
    print("  " + "=" * 50)
    print("  进度:")
    PROGRESS.advance("环境检查")
    setup_proxy(cfg.get("proxy_port", 0))
    check_gh_login()
    init_git(cfg["github_repo_url"])

    PROGRESS.advance("文件准备")
    calib_count = prepare_files(cfg)
    sys.modules[__name__]._calib_count = calib_count

    PROGRESS.advance("参数同步")
    push_to_github(cfg)

    success = wait_and_download(cfg)

    elapsed = format_duration(time.time() - total_start)

    if success:
        PROGRESS.current = PROGRESS.total
        print(f"\n  [████████] 100%  全部完成！  ({elapsed})")
        print()
        script_dir = Path(__file__).parent
        kmodel = next(script_dir.glob("*.kmodel"), None)
        if kmodel:
            print(f"  📦 Kmodel: {kmodel}")
            print(f"  📏 大小:   {format_size(kmodel)}")
            print()
            print("  将此文件拷贝到 K230 开发板即可使用！")
    else:
        url = cfg.get("github_repo_url", "").rstrip(".git") + "/actions"
        print(f"\n  ❌ 转换失败  ({elapsed})")
        print(f"  💡 查看: {url}")

    print()


if __name__ == "__main__":
    main()