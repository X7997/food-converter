#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地 / GitHub Actions 通用 .pt / .onnx -> K230 Kmodel 转换脚本
关键参数支持通过环境变量覆盖，便于在 CI/CD 中调用。
"""

import os
import sys
import time
import glob
from pathlib import Path


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, str(default)))


def _env_list(key: str, default: list) -> list:
    val = os.environ.get(key)
    if val is None:
        return default
    return [int(x.strip()) for x in val.split(",")]


# ========================== 用户配置区（全局变量） ==========================
# 所有配置均支持通过同名环境变量覆盖，方便 GitHub Actions 调用。
# 例: export K230_MODEL_PATH=models/input.pt
MODEL_PATH = _env_str("K230_MODEL_PATH", r"Q:\K230_ultralytics\ultralytics_main\database\runs\detect\results\yolov8n3\weights\best.pt")
KMODEL_PATH = _env_str("K230_KMODEL_PATH", "")
CALIB_IMAGE_DIR = _env_str("K230_CALIB_DIR", r"Q:\K230_ultralytics\ultralytics_main\database\test\images")

INPUT_SHAPE = _env_list("K230_INPUT_SHAPE", [1, 3, 320, 320])
MAX_CALIB_IMAGES = _env_int("K230_MAX_CALIB_IMAGES", 50)

# ONNX 导出参数（仅当 MODEL_PATH 是 .pt 时生效）
ONNX_IMGSZ = _env_int("K230_ONNX_IMGSZ", 320)
ONNX_BATCH = _env_int("K230_ONNX_BATCH", 1)
ONNX_DYNAMIC = os.environ.get("K230_ONNX_DYNAMIC", "false").lower() == "true"
ONNX_SIMPLIFY = os.environ.get("K230_ONNX_SIMPLIFY", "true").lower() == "true"
ONNX_NMS = os.environ.get("K230_ONNX_NMS", "false").lower() == "true"
ONNX_OPSET = _env_int("K230_ONNX_OPSET", 11)
ONNX_HALF = os.environ.get("K230_ONNX_HALF", "false").lower() == "true"
ONNX_VERBOSE = os.environ.get("K230_ONNX_VERBOSE", "true").lower() == "true"

# 量化校准参数
QUANT_TYPE = _env_str("K230_QUANT_TYPE", "uint8")
W_QUANT_TYPE = _env_str("K230_W_QUANT_TYPE", "uint8")
CALIB_METHOD = _env_str("K230_CALIB_METHOD", "Kld")

# 编译选项
TARGET = _env_str("K230_TARGET", "k230")
DUMP_DIR = _env_str("K230_DUMP_DIR", "tmp")
DUMP_IR = os.environ.get("K230_DUMP_IR", "false").lower() == "true"
DUMP_ASM = os.environ.get("K230_DUMP_ASM", "false").lower() == "true"
# ==========================================================================


def log(step: str, msg: str):
    """带时间戳的统一日志输出"""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] [{step}] {msg}")


def setup_env():
    """自动修复 nncase 插件路径，消除本地环境变量未设置的警告"""
    try:
        import nncase_kpu
        plugin_path = os.path.dirname(nncase_kpu.__file__)
        os.environ.setdefault("NNCASE_PLUGIN_PATH", plugin_path)
        log("环境", f"NNCASE_PLUGIN_PATH 已自动设置为: {plugin_path}")
    except ImportError:
        log("环境", "警告: 未找到 nncase_kpu，若后续报错请手动安装 nncase-kpu")


def export_onnx(pt_path: str, imgsz: int = 320) -> str:
    """
    使用 ultralytics 将 .pt 导出为 ONNX。
    返回生成的 .onnx 绝对路径。
    """
    try:
        from ultralytics import YOLO
    except ImportError as e:
        log("错误", f"缺少 ultralytics，无法导出 ONNX: {e}")
        log("提示", "请执行: pip install ultralytics")
        sys.exit(1)

    if not os.path.isfile(pt_path):
        log("错误", f"指定的 .pt 文件不存在: {pt_path}")
        sys.exit(1)

    log("导出", f"正在加载 YOLO 模型: {pt_path}")
    model = YOLO(pt_path)

    log("导出", "开始导出 ONNX（参数见脚本顶部 ONNX_xxx 配置）...")
    success = model.export(
        format="onnx",
        imgsz=imgsz,
        batch=ONNX_BATCH,
        dynamic=ONNX_DYNAMIC,
        simplify=ONNX_SIMPLIFY,
        nms=ONNX_NMS,
        opset=ONNX_OPSET,
        half=ONNX_HALF,
        verbose=ONNX_VERBOSE,
    )

    if not success:
        log("错误", "ultralytics ONNX 导出失败，请查看上方报错")
        sys.exit(1)

    onnx_path = str(Path(pt_path).with_suffix(".onnx"))
    if not os.path.isfile(onnx_path):
        log("错误", f"导出后未找到预期的 ONNX 文件: {onnx_path}")
        sys.exit(1)

    log("导出", f"✅ ONNX 导出成功: {onnx_path}")
    return onnx_path


def resolve_model_path(model_path: str) -> str:
    """
    解析最终的 ONNX 路径：
      - 如果是 .onnx 且存在，直接返回
      - 如果是 .pt 且存在，自动调用 export_onnx 导出后返回 onnx 路径
    """
    if not model_path:
        log("错误", "MODEL_PATH 为空，请在脚本顶部填写模型路径")
        sys.exit(1)

    model_path = os.path.abspath(model_path)
    ext = Path(model_path).suffix.lower()

    if ext == ".onnx":
        if os.path.isfile(model_path):
            log("探测", f"使用指定 ONNX 模型: {model_path}")
            return model_path
        log("错误", f"找不到 ONNX 模型: {model_path}")
        sys.exit(1)

    elif ext == ".pt":
        if os.path.isfile(model_path):
            return export_onnx(model_path, imgsz=ONNX_IMGSZ)
        log("错误", f"指定的 .pt 文件不存在: {model_path}")
        sys.exit(1)

    else:
        log("错误", f"不支持的模型后缀 '{ext}'，请提供 .pt 或 .onnx 文件")
        sys.exit(1)


def resolve_calib_dir(model_path: str, configured_dir: str) -> str:
    """解析校准图目录：优先使用配置，否则尝试模型同级目录下的 images 文件夹"""
    if configured_dir and os.path.isdir(configured_dir):
        return configured_dir

    fallback = os.path.join(os.path.dirname(model_path), "images")
    if os.path.isdir(fallback):
        log("校准", f"使用自动探测到的校准目录: {fallback}")
        return fallback

    return configured_dir or ""


def read_calibration_images(img_dir: str, shape: list, max_num: int):
    """
    读取校准图片并进行预处理。
    返回 list[np.ndarray]，每个元素形状均为 [1, C, H, W]，方便直接喂给 set_tensor_data。
    """
    import cv2
    import numpy as np

    if not img_dir or not os.path.isdir(img_dir):
        raise FileNotFoundError(f"校准图片目录不存在或为空: {img_dir}")

    exts = (".jpg", ".jpeg", ".png", ".bmp")
    files = [f for f in os.listdir(img_dir) if f.lower().endswith(exts)]
    files.sort()

    if not files:
        raise FileNotFoundError(f"在 {img_dir} 中未找到支持的图片格式 {exts}")

    use_count = min(len(files), max_num)
    log("校准", f"目录共 {len(files)} 张图，本次使用前 {use_count} 张作为校准样本")

    data_list = []
    _, C, H, W = shape
    for i, filename in enumerate(files[:use_count], 1):
        img_path = os.path.join(img_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            log("校准", f"  跳过无法读取的图片: {filename}")
            continue

        img = cv2.resize(img, (W, H))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)          # [1, 3, H, W]
        data_list.append(img)

        if i % 10 == 0 or i == use_count:
            log("校准", f"  已加载 {i}/{use_count} 张...")

    if not data_list:
        raise RuntimeError("没有成功加载任何校准图片，请检查目录或图片格式")

    log("校准", f"成功加载 {len(data_list)} 张校准样本，单张形状: {data_list[0].shape}")
    return data_list


def main():
    log("启动", "=" * 60)
    log("启动", "YOLO (.pt/.onnx) -> K230 Kmodel 转换脚本启动")
    log("启动", "=" * 60)

    setup_env()

    try:
        import nncase
        import nncase_kpu
        log("环境", f"nncase 版本检查通过: {getattr(nncase, '__version__', 'unknown')}")
    except ImportError as e:
        log("错误", f"缺少必要依赖: {e}")
        log("提示", "请执行: pip install nncase==2.8.3 nncase-kpu==2.8.3")
        sys.exit(1)

    onnx_file = resolve_model_path(MODEL_PATH)
    if not onnx_file or not os.path.isfile(onnx_file):
        log("错误", "最终仍无法定位 ONNX 模型文件，请检查脚本顶部配置。")
        sys.exit(1)

    if KMODEL_PATH:
        kmodel_file = KMODEL_PATH
    else:
        base = Path(MODEL_PATH).with_suffix("")
        kmodel_file = str(base.with_suffix(".kmodel"))
    log("输出", f"Kmodel 将保存至: {kmodel_file}")

    calib_dir = resolve_calib_dir(onnx_file, CALIB_IMAGE_DIR)
    if not calib_dir:
        log("错误", "未配置有效的校准图片目录，请修改脚本顶部 CALIB_IMAGE_DIR")
        sys.exit(1)

    log("导入", f"正在读取 ONNX 模型: {onnx_file}")
    with open(onnx_file, "rb") as f:
        model_content = f.read()
    log("导入", f"模型大小: {len(model_content) / 1024 / 1024:.2f} MB")

    log("编译", "初始化 nncase 编译器...")
    compile_options = nncase.CompileOptions()
    compile_options.target = TARGET
    compile_options.dump_ir = DUMP_IR
    compile_options.dump_asm = DUMP_ASM
    compile_options.dump_dir = DUMP_DIR

    compiler = nncase.Compiler(compile_options)

    log("导入", "正在将 ONNX 导入为 nncase 计算图...")
    import_options = nncase.ImportOptions()
    compiler.import_onnx(model_content, import_options)
    log("导入", "ONNX 导入成功")

    log("量化", "开始准备 PTQ 量化校准数据...")
    calib_data_list = read_calibration_images(calib_dir, INPUT_SHAPE, MAX_CALIB_IMAGES)

    ptq_options = nncase.PTQTensorOptions()
    ptq_options.samples_count = len(calib_data_list)
    ptq_options.calibrate_method = CALIB_METHOD
    ptq_options.quant_type = QUANT_TYPE
    ptq_options.w_quant_type = W_QUANT_TYPE
    ptq_options.set_tensor_data([calib_data_list])

    log("量化", f"PTQ 参数: method={CALIB_METHOD}, quant_type={QUANT_TYPE}, w_quant_type={W_QUANT_TYPE}, samples={len(calib_data_list)}")
    log("量化", "正在执行 PTQ 量化校准（可能需要几分钟，请耐心等待）...")
    compiler.use_ptq(ptq_options)
    log("量化", "PTQ 量化校准完成")

    log("编译", f"正在编译目标平台 '{TARGET}' 专属模型...")
    compiler.compile()
    log("编译", "编译完成，正在生成二进制...")
    kmodel = compiler.gencode_tobytes()
    log("编译", f"生成成功，模型大小: {len(kmodel) / 1024:.2f} KB")

    os.makedirs(os.path.dirname(kmodel_file) or ".", exist_ok=True)
    with open(kmodel_file, "wb") as f:
        f.write(kmodel)

    log("完成", "=" * 60)
    log("完成", f"✅ 转换成功！Kmodel 已保存: {kmodel_file}")
    log("完成", f"   原始模型: {MODEL_PATH}")
    log("完成", f"   输入尺寸: {INPUT_SHAPE}")
    log("完成", f"   目标平台: {TARGET}")
    log("完成", f"   量化配置: {QUANT_TYPE} (weight: {W_QUANT_TYPE})")
    log("完成", "=" * 60)


if __name__ == "__main__":
    main()
