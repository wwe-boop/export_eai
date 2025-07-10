# export_eai
https://github.com/quic/aimet?tab=readme-ov-file#supported-features
import os
import shlex
import subprocess
import numpy as np

# ------------------------------
# 根据图2提供的工具路径 (必须配置)
# ------------------------------
eai_builder="/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.2.0.1/addons/LPAI/eNPU6/5.7.0/eai_builder/eai_builder"
eai_quantize="/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.2.0.1/addons/LPAI/eNPU6/5.7.0/utilities/tensor_quantization/eai_quantize"
eai_sample_app="/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.2.0.1/addons/LPAI/eNPU6/5.7.0/binaries/linux_x86/eai_sample_app/fixed32/eai_sample_app"
# ------------------------------
# 核心处理函数 (基于图1流程)
# ------------------------------
def run_eai_pipeline(
        eai_model: str,  # EAI模型路径 (.onnx等)
        prep_data_root: str,  # 输入数据目录
        eai_model_dir: str,  # 工作目录
        float_input_name: str,  # 浮点输入文件名
        float_cached_len_name: str,  # 缓存长度文件名
        float_cached_avg_name: str  # 缓存平均值文件名
) -> dict:
    """执行完整的EAI模型量化->推理->反量化流程"""

    # 阶段1: 复制输入数据到工作目录
    for file_name in [float_input_name, float_cached_len_name, float_cached_avg_name]:
        src = os.path.join(prep_data_root, file_name)
        dst = os.path.join(eai_model_dir, file_name)
        cmd = f"cp {src} {dst}"
        result = subprocess.run(shlex.split(cmd), shell=False)
        assert result.returncode == 0, f"复制失败: {cmd}"

    # 阶段2: 量化模型输入
    quant_cmd = (
        f"{eai_quantize} --eai {eai_model} --quantize 1 --input "
        f"{os.path.join(eai_model_dir, float_input_name)} "
        f"{os.path.join(eai_model_dir, float_cached_len_name)} "
        f"{os.path.join(eai_model_dir, float_cached_avg_name)}"
    )
    result = subprocess.run(shlex.split(quant_cmd), shell=False)
    assert result.returncode == 0, f"量化失败: {quant_cmd}"

    # 阶段3: 执行EAI模型推理
    infer_cmd = (
        f"{eai_sample_app} -m {eai_model} -o {eai_model_dir} -i "
        f"{os.path.join(eai_model_dir, 'quantized_' + float_input_name)} "
        f"{os.path.join(eai_model_dir, 'quantized_' + float_cached_len_name)} "
        f"{os.path.join(eai_model_dir, 'quantized_' + float_cached_avg_name)}"
    )
    result = subprocess.run(shlex.split(infer_cmd), shell=False)
    assert result.returncode == 0, f"推理失败: {infer_cmd}"

    # 阶段4: 反量化模型输出
    dequant_cmd = (
        f"{eai_quantize} --eai {eai_model} --quantize 0 --output "
        f"{os.path.join(eai_model_dir, 'output_0.raw')} "
        f"{os.path.join(eai_model_dir, 'output_1.raw')} "
        f"{os.path.join(eai_model_dir, 'output_2.raw')}"  # 修正原始代码的重复行
    )
    result = subprocess.run(shlex.split(dequant_cmd), shell=False)
    assert result.returncode == 0, f"反量化失败: {dequant_cmd}"

    # 阶段5: 读取处理结果
    results = {}
    for i in range(3):
        raw_path = os.path.join(eai_model_dir, f'dequantized_output_{i}.raw')
        results[f'output_{i}'] = np.fromfile(raw_path, dtype=np.float32)

    return results


# ------------------------------
# 使用示例
# ------------------------------
if __name__ == "__main__":
    # 用户需要配置以下参数
    output_data = run_eai_pipeline(
        eai_model="/path/to/your_model.onnx",
        prep_data_root="./input_data",
        eai_model_dir="./workspace",
        float_input_name="input.bin",
        float_cached_len_name="seq_len.bin",
        float_cached_avg_name="avg_values.bin"
    )

    print(f"获取输出张量: {list(output_data.keys())}")
    print(f"Output0 shape: {output_data['output_0'].shape}")

    # 1.
    # encoding的被操作，什么差距，如何生成准确值的encoding，解决encoding，
    #
    # 2.
    # 真实音频量化（不太完整）然后比较相似度，量化的这个需要
    #
    # 3.
    # eai模型推理比较
    #
    # 4.
    # 输入改到40
