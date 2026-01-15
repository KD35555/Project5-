import streamlit as st
import numpy as np
import os
from PIL import Image
from dinov2_numpy import Dinov2Numpy
from preprocess_image import resize_short_side

# 1. 页面基础设置 (标题、图标、布局)
st.set_page_config(
    page_title="AI 图像检索系统",
    page_icon="🔍",
    layout="wide"
)

# 2. 侧边栏：设置区域
with st.sidebar:
    st.header("⚙️ 系统设置")
    top_k = st.slider("显示相似图片数量 (Top K)", min_value=1, max_value=20, value=8)
    st.info("💡 提示：请确保你已经运行过 'step2_build_index.py' 建立了索引库。")
    st.markdown("---")
    st.markdown("**Core Model:** Vision Transformer (ViT)")
    st.markdown("**Backbone:** DINOv2 Base")

# 3. 主界面：标题
st.title("🔍 AI 以图搜图系统 (Image Retrieval)")
st.markdown("""
<style>
    .big-font { font-size:20px !important; color: gray; }
</style>
""", unsafe_allow_html=True)
st.markdown('<p class="big-font">上传一张图片，系统将从图库中找出最相似的结果。</p>', unsafe_allow_html=True)

# 4. 加载模型和索引 (使用缓存，只加载一次，速度快)
@st.cache_resource
def load_system():
    # 加载模型
    if not os.path.exists("vit-dinov2-base.npz"):
        return None, None, "❌ 错误：找不到模型文件 vit-dinov2-base.npz"
    
    weights = np.load("vit-dinov2-base.npz")
    model = Dinov2Numpy(weights)
    
    # 加载索引
    if not os.path.exists("index_features.npy") or not os.path.exists("index_paths.npy"):
        return None, None, "❌ 错误：找不到索引文件！请先运行 step2_build_index.py"
    
    gallery_feats = np.load("index_features.npy")
    gallery_paths = np.load("index_paths.npy")
    
    return model, (gallery_feats, gallery_paths), "OK"

# 显示加载状态
with st.spinner('正在启动 AI 引擎...'):
    vit_model, index_data, status_msg = load_system()

if status_msg != "OK":
    st.error(status_msg)
    st.stop()

gallery_features, gallery_paths = index_data

# 5. 上传图片区域
uploaded_file = st.file_uploader("📂 请把图片拖拽到这里，或点击上传", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # --- 布局：左边显示原图，右边显示结果 ---
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🖼️ 你上传的图片")
        # 显示用户上传的图
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True)
        
        # 保存临时文件用于处理
        temp_path = "temp_query.jpg"
        image.save(temp_path)

    # --- 开始搜索 ---
    with col2:
        st.subheader(f"🚀 搜索结果 (Top {top_k})")
        
        # 1. 预处理 & 推理
        try:
            query_tensor = resize_short_side(temp_path)
            query_feat = vit_model(query_tensor) # (1, 768)
        except Exception as e:
            st.error(f"处理图片出错: {e}")
            st.stop()

        # 2. 计算相似度 (矩阵乘法)
        similarity = gallery_features @ query_feat.T # (N, 1)
        similarity = similarity.flatten()

        # 3. 排序
        indices = np.argsort(similarity)[-top_k:][::-1]

        # 4. 展示结果 (网格布局)
        # 比如每行显示 4 张图
        cols_per_row = 4
        rows = [st.columns(cols_per_row) for _ in range((top_k + cols_per_row - 1) // cols_per_row)]
        
        for i, idx in enumerate(indices):
            row_idx = i // cols_per_row
            col_idx = i % cols_per_row
            
            score = similarity[idx]
            path = gallery_paths[idx]
            
            # 显示图片和相似度
            with rows[row_idx][col_idx]:
                # 检查图片是否存在
                if os.path.exists(path):
                    st.image(path, caption=f"相似度: {score:.4f}", use_container_width=True)
                else:
                    st.warning(f"图片丢失: {path}")

    # 清理临时文件
    if os.path.exists(temp_path):
        os.remove(temp_path)