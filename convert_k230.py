#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
云端转换核心脚本 - 在 GitHub Actions 中运行
所有参数通过环境变量传入，本地无需修改此文件。
"""

import os
import sys
import time
import glob
from pathlib import Path


def _env_str(key, default):
    return os.environ.get(key, default)


def _env_int(key, default):
    return int(os.environ.get(key, str(default)))


def _env_list(key, default):
    val = os.environ.get(key)
    if val is None:
        return default
    return [int(x.strip()) for x in val.split(",")]


# ─── 环境变量驱动配置（由 easy_k230.py / workflow 自动设置）──
MODEL_PATH = _env_str("K230_MODEL_PATH", "")
KMODEL_PATH = _env_str("K230_KMODEL_PATH", "")
CALIB_IMAGE_DIR = _env_str("K230_CALIB_DIR", "")
INPUT_SHAPE = _env_list("K230_INPUT_SHAPE", [1, 3, 320, 320])
MAX_CALIB_IMAGES = _env_int("K230_MAX_CALIB_IMAGES", 999)
ONNX_IMGSZ = _env_int("K230_ONNX_IMGSZ", 320)
ONNX_BATCH = _env_int("K230_ONNX_BATCH", 1)
ONNX_DYNAMIC = os.environ.get("K230_ONNX_DYNAMIC", "false").lower() == "true"
ONNX_SIMPLIFY = os.environ.get("K230_ONNX_SIMPLIFY", "true").lower() == "true"
ONNX_NMS = os.environ.get("K230_ONNX_NMS", "false").lower() == "true"
ONNX_OPSET = _env_int("K230_ONNX_OPSET", 11)
ONNX_HALF = os.environ.get("K230_ONNX_HALF", "false").lower() == "true"
ONNX_VERBOSE = os.environ.get("K230_ONNX_VERBOSE", "true").lower() == "true"
QUANT_TYPE = _env_str("K230_QUANT_TYPE", "uint8")
W_QUANT_TYPE = _env_str("K230_W_QUANT_TYPE", "uint8")
CALIB_METHOD = _env_str("K230_CALIB_METHOD", "Kld")
INPUT_TYPE = _env_str("K230_INPUT_TYPE", "uint8")
OUTPUT_TYPE = _env_str("K230_OUTPUT_TYPE", "float16")
TARGET = _env_str("K230_TARGET", "k230")
DUMP_DIR = _env_str("K230_DUMP_DIR", "tmp")
DUMP_IR = os.environ.get("K230_DUMP_IR", "false").lower() == "true"
DUMP_ASM = os.environ.get("K230_DUMP_ASM", "false").lower() == "true"


def log(step, msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [{step}] {msg}")


def patch_onnx_input_normalize(onnx_path):
    """
    修复 K230 uint8 输入问题：在 ONNX 图开头显式添加 /255.0 归一化。
    
    问题：YOLO 模型训练时已归一化到 [0,1]，ONNX 期望 float32 [0,1] 输入。
         nncase 的 input_type=uint8 算子折叠不可靠。
    解决：在 ONNX 图中插入 Div(255.0)，使模型接受 float32 [0,255] 输入。
         kmodel 转换时使用 input_type=float32，避免 nncase 添加额外 dequantize。
    """
    import onnx
    from onnx import helper

    model = onnx.load(onnx_path)
    graph = model.graph
    
    old_input_name = graph.input[0].name
    new_input_name = old_input_name + "_norm"
    
    # 插入 Constant(255.0) 节点
    const_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["div_const_255"],
        value=helper.make_tensor(
            name="const_255_val",
            data_type=onnx.TensorProto.FLOAT,
            dims=[1],
            vals=[255.0],
        ),
    )
    
    # 插入 Div 节点: normalized = input / 255.0
    div_node = helper.make_node(
        "Div",
        inputs=[old_input_name, "div_const_255"],
        outputs=[new_input_name],
    )
    
    # 插入到图开头
    graph.node.insert(0, const_node)
    graph.node.insert(1, div_node)
    
    # 将后续节点中引用原输入名的地方替换为新名称
    for node in graph.node[2:]:
        for i in range(len(node.input)):
            if node.input[i] == old_input_name:
                node.input[i] = new_input_name
    
    onnx.save(model, onnx_path)
    log("补丁", f"✅ ONNX 归一化补丁完成: {old_input_name} -> Div(255) -> {new_input_name}")
    return onnx_path


def setup_env():
    try:
        import nncase_kpu
        os.environ.setdefault("NNCASE_PLUGIN_PATH", os.path.dirname(nncase_kpu.__file__))
    except ImportError:
        pass


def export_onnx(pt_path, imgsz=320):
    from ultralytics import YOLO
    log("导出", f"加载模型: {pt_path}")
    model = YOLO(pt_path)
    log("导出", f"导出 ONNX (imgsz={imgsz})...")
    success = model.export(
        format="onnx", imgsz=imgsz, batch=ONNX_BATCH,
        dynamic=ONNX_DYNAMIC, simplify=ONNX_SIMPLIFY,
        nms=ONNX_NMS, opset=ONNX_OPSET, half=ONNX_HALF, verbose=ONNX_VERBOSE,
    )
    if not success:
        log("错误", "ONNX 导出失败")
        sys.exit(1)
    onnx_path = str(Path(pt_path).with_suffix(".onnx"))
    log("导出", f"✅ ONNX 导出成功: {onnx_path}")
    return onnx_path


def read_calibration_images(img_dir, shape, max_num):
    import cv2
    import numpy as np

    exts = (".jpg", ".jpeg", ".png", ".bmp")
    files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(exts))
    if not files:
        raise FileNotFoundError(f"校准图目录为空: {img_dir}")

    use_count = min(len(files), max_num)
    log("校准", f"共 {len(files)} 张，使用 {use_count} 张")

    # ════════════════════════════════════════════════════════════════════════════
    # 预处理参数（K230 部署时必须对齐！）
    # ─────────────────────────────────────────────────────────────────────────────
    # 1. 输入范围：保持 0~255（float32），ONNX 中显式 /255.0 归一化
    # 2. 颜色格式：RGB（cv2.cvtColor BGR→RGB）
    # 3. Layout：NCHW（通道在前）
    # 4. 均值/方差：无（YOLO 默认不使用 mean/std 归一化）
    # 5. 输入尺寸：由 INPUT_SHAPE 指定，当前为 [1, 3, 320, 320]
    # 
    # K230 部署代码必须执行相同的预处理：
    #    - ai2d resize 后，输出 float32 [0,255]（不要做归一化）
    #    - ONNX 模型内部已有 /255.0 → kmodel 内部自动处理归一化
    #    - 确保输入是 RGB 格式（不是 BGR）
    #    - 模型输出为 float16，后处理使用 float 计算
    # ════════════════════════════════════════════════════════════════════════════
    log("校准", "预处理: RGB格式, NCHW布局, float32范围[0,255], 输出float16")

    data_list = []
    _, C, H, W = shape
    for i, filename in enumerate(files[:use_count], 1):
        img = cv2.imread(os.path.join(img_dir, filename))
        if img is None:
            continue
        img = cv2.resize(img, (W, H))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
        data_list.append(img)
        if i % 50 == 0:
            log("校准", f"  已加载 {i}/{use_count}")

    log("校准", f"成功加载 {len(data_list)} 张, 形状 {data_list[0].shape}")
    return data_list


def main():
    log("启动", "=" * 50)
    log("启动", "YOLO -> K230 Kmodel 云端转换")
    log("启动", f"输入尺寸={INPUT_SHAPE}  目标={TARGET}  量化={QUANT_TYPE}/{W_QUANT_TYPE}  input={INPUT_TYPE}  output={OUTPUT_TYPE}")

    setup_env()

    import nncase
    import nncase_kpu

    if not MODEL_PATH:
        log("错误", "K230_MODEL_PATH 未设置")
        sys.exit(1)

    ext = Path(MODEL_PATH).suffix.lower()
    if ext == ".pt":
        onnx_file = export_onnx(MODEL_PATH, imgsz=ONNX_IMGSZ)
    elif ext == ".onnx":
        onnx_file = MODEL_PATH
    else:
        log("错误", f"不支持的模型格式: {ext}")
        sys.exit(1)

    if not os.path.isfile(onnx_file):
        log("错误", f"ONNX 文件不存在: {onnx_file}")
        sys.exit(1)
    
    # ★ 修复 K230 uint8 问题：显式添加 /255.0 归一化到 ONNX 图中
    onnx_file = patch_onnx_input_normalize(onnx_file)

    with open(onnx_file, "rb") as f:
        model_content = f.read()
    log("导入", f"ONNX 大小: {len(model_content) / 1024 / 1024:.1f} MB")

    compile_options = nncase.CompileOptions()
    compile_options.target = TARGET
    compile_options.dump_ir = DUMP_IR
    compile_options.dump_asm = DUMP_ASM
    compile_options.dump_dir = DUMP_DIR

    compiler = nncase.Compiler(compile_options)
    compiler.import_onnx(model_content, nncase.ImportOptions())
    log("导入", "ONNX 导入成功")

    if not CALIB_IMAGE_DIR:
        log("错误", "K230_CALIB_DIR 未设置")
        sys.exit(1)

    calib_data = read_calibration_images(CALIB_IMAGE_DIR, INPUT_SHAPE, MAX_CALIB_IMAGES)

    ptq = nncase.PTQTensorOptions()
    ptq.samples_count = len(calib_data)
    ptq.calibrate_method = CALIB_METHOD
    ptq.quant_type = QUANT_TYPE
    ptq.w_quant_type = W_QUANT_TYPE
    ptq.input_type = INPUT_TYPE
    ptq.output_type = OUTPUT_TYPE
    ptq.set_tensor_data([calib_data])
    log("量化", f"PTQ: method={CALIB_METHOD} quant={QUANT_TYPE} w_quant={W_QUANT_TYPE} input={INPUT_TYPE} output={OUTPUT_TYPE} samples={len(calib_data)}")
    compiler.use_ptq(ptq)
    log("量化", "PTQ 校准完成")

    log("编译", f"编译 K230 模型...")
    compiler.compile()
    kmodel = compiler.gencode_tobytes()
    log("编译", f"生成成功: {len(kmodel) / 1024:.1f} KB")

    os.makedirs(os.path.dirname(KMODEL_PATH) or "results", exist_ok=True)
    with open(KMODEL_PATH, "wb") as f:
        f.write(kmodel)

    log("完成", f"✅ 已保存: {KMODEL_PATH}")


if __name__ == "__main__":
    main()