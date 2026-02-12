
import os
import torch
import numpy as np
import huggingface_hub
from tabpfn import TabPFNClassifier

def main():
    print("="*50)
    print("TabPFN 模型参数量统计脚本")
    print("="*50)

    # ---------------------------------------------------------
    # 1. Hugging Face 认证 (解决你遇到的 RuntimeError)
    # ---------------------------------------------------------
    # 如果你已经配置了环境变量 HF_TOKEN，这里可以跳过。
    # 否则，请将下面的 "YOUR_HF_TOKEN" 替换为你从 Hugging Face 获取的 Token。
    hf_token = os.getenv("HF_TOKEN")
    
    if hf_token is None:
        print("[提示] 未检测到 HF_TOKEN 环境变量。")
        print("如果你尚未登录 Hugging Face，请在下方代码中填入 Token，或在终端运行 `huggingface-cli login`。")
        # 可以在这里硬编码 Token 进行临时测试：
        # huggingface_hub.login(token="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    else:
        print("[Info] 检测到环境变量 HF_TOKEN，尝试自动登录...")
        try:
            huggingface_hub.login(token=hf_token)
            print("[Success] Hugging Face 登录成功！")
        except Exception as e:
            print(f"[Warning] 登录失败: {e}")

    # ---------------------------------------------------------
    # 2. 准备 Dummy 数据
    # ---------------------------------------------------------
    print("\n[Step 1] 准备测试数据...")
    # 模拟一个简单的二分类任务，TabPFN 不需要大量数据即可初始化
    X_train = np.random.rand(50, 10)  # 50个样本，10个特征
    y_train = np.random.randint(0, 2, 50) # 二分类标签
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ---------------------------------------------------------
    # 3. 初始化并加载 TabPFN
    # ---------------------------------------------------------
    print("[Step 2] 初始化 TabPFNClassifier...")
    # TabPFN v2 在 fit 时会下载/加载模型权重
    classifier = TabPFNClassifier(device=device)
    
    print("[Step 3] 正在加载模型 (fit)...")
    try:
        classifier.fit(X_train, y_train)
    except Exception as e:
        print("\n" + "!"*50)
        print("模型加载失败！通常是因为没有 Hugging Face 权限。")
        print("请确保你已在 Hugging Face 网站同意 TabPFN-2.5 的协议，并正确设置了 Token。")
        print("报错详情:", e)
        print("!"*50 + "\n")
        return

    # ---------------------------------------------------------
    # 4. 统计参数量
    # ---------------------------------------------------------
    print("[Step 4] 统计参数量...")
    
    # TabPFNClassifier 内部通常有一个 model 属性存储实际的 PyTorch 模型
    # 我们尝试通过反射找到它
    model_instance = None
    
    if hasattr(classifier, 'model'):
        model_instance = classifier.model
    elif hasattr(classifier, 'classifier'): # 有些版本可能是 classifier
        model_instance = classifier.classifier
    
    if model_instance is not None and isinstance(model_instance, torch.nn.Module):
        total_params = sum(p.numel() for p in model_instance.parameters())
        trainable_params = sum(p.numel() for p in model_instance.parameters() if p.requires_grad)
        
        print(f"\n{'-'*30}")
        print(f"TabPFN Model Structure: {type(model_instance).__name__}")
        print(f"Total Parameters:     {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Model Size (approx):  {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
        print(f"{'-'*30}\n")
        
        # 打印具体的层信息（可选）
        # print(model_instance)
        
    else:
        print("[Warning] 无法直接访问内部 PyTorch 模型对象，尝试通用搜索...")
        # 暴力搜索一下属性
        found = False
        for attr_name in dir(classifier):
            attr = getattr(classifier, attr_name)
            if isinstance(attr, torch.nn.Module):
                print(f"Found internal module: {attr_name}")
                total_params = sum(p.numel() for p in attr.parameters())
                print(f"Parameters in '{attr_name}': {total_params:,}")
                found = True
                break
        
        if not found:
            print("无法找到底层的 torch.nn.Module 对象，无法统计参数。")

if __name__ == "__main__":
    main()
