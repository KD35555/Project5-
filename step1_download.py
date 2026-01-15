import os
import requests
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor # 导入多线程工具

def download_single_image(args):
    """下载单张图片的函数，专门给线程调用"""
    index, url, save_folder = args
    save_path = os.path.join(save_folder, f"img_{index}.jpg")
    
    # 1. 断点续传：如果文件已经存在且大小不为0，就跳过
    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        return True

    try:
        # 2. 设置超时：2秒没反应直接跳过，为了速度
        response = requests.get(url, timeout=2)
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(response.content)
            return True
    except:
        # 下载失败（链接失效）是常事，直接忽略
        pass
    return False

def download_images_fast(csv_path, target_count=20000, save_folder="static/gallery"):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # 1. 读取 CSV
    print(f"正在读取 {csv_path} ...")
    try:
        df = pd.read_csv(csv_path)
        # 智能查找 URL 列
        url_col = [c for c in df.columns if 'url' in c.lower()]
        if url_col:
            urls = df[url_col[0]].tolist()
        else:
            urls = df.iloc[:, 0].tolist()
    except Exception as e:
        print(f"❌ CSV 读取失败: {e}")
        return

    # 2. 设置扫描范围
    # 为了凑够 20,000 张，我们尝试读取前 35,000 个链接（预留死链空间）
    scan_limit = 35000 
    urls_to_download = urls[:scan_limit] 
    print(f"准备在前 {len(urls_to_download)} 个链接中下载有效图片...")

    # 3. 打包任务
    tasks = []
    for i, url in enumerate(urls_to_download):
        tasks.append((i, url, save_folder))

    # 4. 多线程极速下载 (32个工人同时搬砖)
    print(f"🚀 启动 32 线程极速下载...")
    
    with ThreadPoolExecutor(max_workers=32) as executor:
        # 使用 tqdm 显示进度条
        results = list(tqdm(executor.map(download_single_image, tasks), total=len(tasks), unit="img"))

    # 5. 统计结果
    success_count = sum(results)
    print("-" * 30)
    print(f"✅ 下载结束！")
    print(f"成功下载数量: {success_count} 张")
    print(f"保存位置: {save_folder}")
    print("-" * 30)
    
    if success_count < target_count:
        print(f"⚠️ 提示：只下载了 {success_count} 张。")
        print("如果觉得不够，请把代码里的 scan_limit = 35000 改得更大。")

if __name__ == "__main__":
    if os.path.exists("data.csv"):
        # 这里的 target_count 只是用来提示显示的，实际取决于 scan_limit
        download_images_fast("data.csv", target_count=20000)
    else:
        print("❌ 错误：找不到 data.csv 文件，请检查路径！")