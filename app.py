from pathlib import Path
from PIL import Image
import streamlit as st
import config
from utils import load_model, infer_uploaded_image, infer_uploaded_video, infer_uploaded_webcam

# 设置页面布局和主题样式
st.set_page_config(
    page_title="多气象环境下基于YOLO的多目标检测与追踪系统",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS以优化界面样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #374151;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-divider {
        background-color: #e5e7eb;
        height: 2px;
        margin: 1.5rem 0;
        border-radius: 1px;
    }
    .sidebar-section {
        margin-bottom: 1.5rem;
        padding: 1rem;
        background-color: #f8fafc;
        border-radius: 0.5rem;
    }
    .sidebar-section-header {
        font-weight: 600;
        margin-bottom: 0.75rem;
        color: #1e3a8a;
    }
    .stButton > button {
        background-color: #1e3a8a;
        color: white;
        border-radius: 0.375rem;
    }
    .stButton > button:hover {
        background-color: #1e40af;
    }
    .stAlert {
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# 主页面标题
col_title = st.columns(1)
with col_title[0]:
    st.markdown('<h1 class="main-header">多气象环境下基于YOLO的多目标检测与追踪系统</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">支持图像、视频和摄像头实时检测与跟踪</p>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# 侧边栏配置区域
with st.sidebar:
    # 模型配置区域
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-header">模型配置</div>', unsafe_allow_html=True)
    
    # 任务选项 - 直接使用中文选项避免format_func可能导致的问题
    task_type = st.selectbox(
        "选择任务类型",
        ["检测", "跟踪"],
        index=0,  # 设置默认值为检测
        help="选择要执行的任务类型"
    )
    
    # 将中文任务类型转换为内部使用的英文标识
    task_id = "Detection" if task_type == "检测" else "Tracking"
    
    # 模型选择
    model_type = st.selectbox(
        "选择模型",
        config.DETECTION_MODEL_LIST,
        index=0,  # 设置默认值为列表的第一个模型
        help="选择要使用的YOLOv8模型"
    )
    
    # 跟踪选项
    is_tracking = task_type == "跟踪"
    tracker_type = None
    if is_tracking:
        tracker_type = st.selectbox(
            "选择跟踪器",
            config.TRACKER_TYPE_LIST,
            index=0,  # 设置默认值为列表的第一个跟踪器
            help="选择目标跟踪算法"
        )
    
    # 置信度阈值（跟踪模式下使用较低的阈值以允许更多检测）
    if is_tracking:
        # 跟踪模式下默认置信度较低
        confidence = float(st.slider(
            "模型置信度阈值", 
            min_value=10, 
            max_value=100, 
            value=30,
            help="跟踪模式建议使用较低的置信度值"
        )) / 100
    else:
        confidence = float(st.slider(
            "模型置信度阈值", 
            min_value=30, 
            max_value=100, 
            value=50,
            help="检测模式建议使用适中的置信度值"
        )) / 100
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 数据源配置区域
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-header">数据源配置</div>', unsafe_allow_html=True)
    
    # 根据任务类型显示不同的数据源选项
    if is_tracking:
        # 跟踪模式下只显示视频和摄像头
        available_sources = ["Video", "Webcam"]
        source_options = ["视频", "摄像头"]
        default_index = 0  # 默认选择视频
        st.info("跟踪模式仅支持视频和摄像头输入")
    else:
        # 检测模式下显示所有数据源
        available_sources = config.SOURCES_LIST
        source_options = ["图像", "视频", "摄像头"]
        default_index = 0  # 默认选择图像
    
    # 使用直接的选项映射，避免format_func导致的空框问题
    source_selectbox = st.selectbox(
        "选择数据源",
        source_options,
        index=default_index,
        help="选择要处理的数据源类型"
    )
    
    # 将显示的中文选项映射回内部使用的英文标识
    source_mapping = {"图像": "Image", "视频": "Video", "摄像头": "Webcam"}
    source = source_mapping[source_selectbox]
    
    st.markdown('</div>', unsafe_allow_html=True)

# 加载预训练模型
model = None
try:
    model_path = Path(config.DETECTION_MODEL_DIR, str(model_type))
    with st.spinner("正在加载模型，请稍候..."):
        model = load_model(model_path)
    st.success(f"成功加载模型: {model_type}")
except Exception as e:
        st.error(f"加载模型失败: {str(e)}")
        st.info("请检查模型文件路径是否正确")

# 根据选择的数据源执行相应的推理函数
if model is not None:
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    if source == "Image":
        if is_tracking:
            st.warning("跟踪功能不适用于图像。请选择'检测'任务或'视频'/'摄像头'源。")
        infer_uploaded_image(confidence, model, is_tracking, tracker_type)
    elif source == "Video":
        infer_uploaded_video(confidence, model, is_tracking, tracker_type)
    elif source == "Webcam":
        infer_uploaded_webcam(confidence, model, is_tracking, tracker_type)
else:
    # 显示引导信息
    if not model_type:
        st.markdown("### 操作指南")
        st.markdown("1. 在侧边栏选择任务类型（检测/跟踪）")
        st.markdown("2. 选择要使用的YOLOv8模型")
        st.markdown("3. 调整置信度阈值")
        st.markdown("4. 选择数据源（图像/视频/摄像头）")
        st.markdown("5. 上传文件并点击'执行'按钮开始处理")