import os
import numpy as np
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
from dinov2_numpy import Dinov2Numpy
from preprocess_image import resize_short_side

# ==========================================
# 核心：定义一个处理“一批”图片的函数
# ==========================================
def process_batch(image_paths):
    # 每个进程独立加载一次模型
    # 注意：这里我们不需要 try-except 包裹模型加载，
    # 如果模型文件坏了，直接报错让我们知道反而更好。
    if not os.path.exists("vit-dinov2-base.npz"):
        return [], []
        
    weights = np.load("vit-dinov2-base.npz")
    vit = Dinov2Numpy(weights)

    batch_features = []
    batch_paths = []

    for path in image_paths:
        try:
            # 1. 快速检查文件大小，跳过损坏的小文件 (<1KB)
            if os.path.getsize(path) < 1024: 
                continue

            # 2. 预处理
            input_tensor = resize_short_side(path)
            
            # 3. 模型推理
            feature = vit(input_tensor)
            
            # 4. 收集结果
            batch_features.append(feature)
            batch_paths.append(path)
        except:
            # 遇到任何坏图直接跳过，不报错
            continue
            
    return batch_features, batch_paths

# ==========================================
# 主程序
# ==========================================
def build_index_fast(image_folder="static/gallery"):
    # 1. 扫描所有图片
    print("正在扫描图片文件...")
    all_paths = sorted(glob(os.path.join(image_folder, "*.jpg")))
    total_imgs = len(all_paths)
    
    if total_imgs == 0:
        print("❌ 没找到图片，请先运行 step1 下载！")
        return

    print(f"找到 {total_imgs} 张图片。")
    print(f"🚀 启动多进程加速计算...")

    # 2. 将图片分成很多小批次 (每批 100 张)
    batch_size = 100
    chunks = [all_paths[i:i + batch_size] for i in range(0, total_imgs, batch_size)]

    all_features = []
    valid_paths = []

    # 3. 启动进程池
    # ====================================================
    # 🔥 关键修改：强制设置为 4 个进程，防止电脑卡死
    # ====================================================
    num_processes = 4 
    print(f"已启动 {num_processes} 个稳定进程同时计算 (请耐心等待1-2分钟预热)...")

    # 使用 if __name__ 保护是 Windows 下多进程的硬性要求
    with Pool(processes=num_processes) as pool:
        # 使用 tqdm 显示进度条
        for features, paths in tqdm(pool.imap(process_batch, chunks), total=len(chunks), unit="batch"):
            if len(features) > 0:
                all_features.extend(features)
                valid_paths.extend(paths)

    # 4. 整合保存
    print("正在整合数据并保存...")
    if len(all_features) > 0:
        final_features = np.concatenate(all_features, axis=0)
        final_paths = np.array(valid_paths)
        
        np.save("index_features.npy", final_features)
        np.save("index_paths.npy", final_paths)
        
        print("-" * 30)
        print(f"✅ 索引构建完成！")
        print(f"成功处理: {len(final_paths)} / {total_imgs} 张图片")
        print(f"特征文件: index_features.npy {final_features.shape}")
        print("-" * 30)
    else:
        print("❌ 失败：没有生成任何特征，可能是图片全部损坏。")

if __name__ == "__main__":
    build_index_fast()