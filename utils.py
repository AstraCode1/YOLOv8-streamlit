#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   @File Name:     utils.py
   @Author:        Luyao.zhang
   @Date:          2023/5/16
   @Description: 工具函数，包含模型加载、推理和跟踪功能
-------------------------------------------------
"""
from ultralytics import YOLO
import streamlit as st
import cv2
from PIL import Image
import tempfile
import numpy as np


def _display_detected_frames(conf, model, st_frame, image, is_tracking=False, tracker_type=None):
    """
    在视频帧上显示检测到的对象，使用YOLOv8模型，可选跟踪功能。
    
    :param conf (float): 对象检测的置信度阈值。
    :param model (YOLOv8): 包含YOLOv8模型的实例。
    :param st_frame (Streamlit object): 用于显示检测视频的Streamlit对象。
    :param image (numpy array): 表示视频帧的numpy数组。
    :param is_tracking (bool): 是否启用对象跟踪。
    :param tracker_type (str): 要使用的跟踪器类型（bytetrack或botsort）。
    :return: None
    """
    # 调整图像到标准大小
    image = cv2.resize(image, (720, int(720 * (9 / 16))))

    # 使用YOLOv8模型预测或跟踪图像中的对象
    try:
        if is_tracking:
            # 跟踪模式使用较低的置信度阈值并保持跟踪
            tracker_config = f"源代码/ultralytics/cfg/trackers/{tracker_type}.yaml" if tracker_type else "源代码/ultralytics/cfg/trackers/bytetrack.yaml"
            res = model.track(image, conf=conf, persist=True, tracker=tracker_config)
        else:
            # 仅检测模式
            res = model.predict(image, conf=conf)

        # 获取检测结果
        boxes = res[0].boxes
        class_names = model.names
        
        # 收集并统计检测结果
        class_counts = {}
        total_objects = len(boxes)
        
        for box in boxes:
            class_id = int(box.cls)
            class_name = class_names[class_id]
            if class_name in class_counts:
                class_counts[class_name] += 1
            else:
                class_counts[class_name] = 1
        
        # 在视频帧上绘制检测到的对象
        res_plotted = res[0].plot()
        
        # 创建一个包含视频和统计信息的容器
        cols = st_frame.columns([3, 1])  # 3:1的比例分配空间给视频和统计信息
        
        with cols[0]:
            st.image(res_plotted,
                     caption='跟踪结果' if is_tracking else '检测结果',
                     channels="BGR",
                     use_column_width=True
                     )
        
        with cols[1]:
            # 使用卡片样式显示统计信息
            st.markdown('<div style="background-color: #f8fafc; padding: 1rem; border-radius: 0.5rem;">', unsafe_allow_html=True)
            st.markdown(f"### 实时{is_tracking and '跟踪' or '检测'}统计")
            st.markdown(f"**总检测数**: {total_objects}")
            
            if class_counts:
                st.markdown("**类别分布**:")
                for class_name, count in class_counts.items():
                    st.markdown(f"- {class_name}: {count}")
            else:
                st.info("当前帧未检测到任何对象")
            
            # 显示置信度阈值
            st.markdown(f"**置信度阈值**: {conf:.2f}")
            
            # 如果是跟踪模式，显示跟踪器类型
            if is_tracking:
                st.markdown(f"**跟踪器**: {tracker_type}")
            st.markdown('</div>', unsafe_allow_html=True)
    except Exception as e:
        st_frame.error(f"处理帧时出错: {str(e)}")
        st_frame.image(image, channels="BGR", use_column_width=True)


@st.cache_resource
def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model


def infer_uploaded_image(conf, model, is_tracking=None, tracker_type=None):
    """
    对上传的图像执行推理
    :param conf: YOLOv8模型的置信度
    :param model: 包含YOLOv8模型的实例
    :param is_tracking: 是否启用跟踪（不适用于图像）
    :param tracker_type: 使用的跟踪器类型（不适用于图像）
    :return: None
    """
    # 在主内容区域显示文件上传器，而不是侧边栏
    st.markdown("### 图像检测")
    st.markdown("请上传一张图像，然后点击'执行检测'按钮进行处理")
    
    source_img = st.file_uploader(
        label="选择图像文件",
        type=("jpg", "jpeg", "png", 'bmp', 'webp'),
        help="支持的图像格式: JPG, JPEG, PNG, BMP, WebP"
    )

    # 创建两列布局
    if source_img:
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            # 添加边框和阴影效果的容器
            st.markdown('<div style="border: 1px solid #e5e7eb; border-radius: 0.5rem; padding: 0.5rem; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);">', unsafe_allow_html=True)
            uploaded_image = Image.open(source_img)
            st.image(
                image=uploaded_image,
                caption="原始图像",
                use_column_width=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

        # 执行检测按钮
        execute_btn = st.button("执行检测", use_container_width=True)
        
        if execute_btn:
            with st.spinner("正在处理图像，请稍候..."):
                try:
                    # 执行模型预测
                    res = model.predict(uploaded_image, conf=conf)
                    boxes = res[0].boxes
                    res_plotted = res[0].plot()[:, :, ::-1]  # 转换为RGB格式

                    with col2:
                        # 显示检测结果图像
                        st.markdown('<div style="border: 1px solid #e5e7eb; border-radius: 0.5rem; padding: 0.5rem; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);">', unsafe_allow_html=True)
                        st.image(
                            res_plotted,
                            caption="检测结果",
                            use_column_width=True
                        )
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # 获取类别名称和置信度信息
                    class_names = model.names
                    detections = []
                    class_counts = {}
                    
                    # 收集所有检测结果
                    for i, box in enumerate(boxes):
                        class_id = int(box.cls)
                        class_name = class_names[class_id]
                        confidence = float(box.conf)
                        x, y, w, h = box.xywh[0].tolist()
                        
                        detections.append({
                            "id": i + 1,
                            "class": class_name,
                            "confidence": confidence,
                            "coordinates": {"x": x, "y": y, "width": w, "height": h}
                        })
                        
                        # 统计每个类别的数量
                        if class_name in class_counts:
                            class_counts[class_name] += 1
                        else:
                            class_counts[class_name] = 1
                    
                    # 使用可折叠面板显示详细结果
                    with st.expander("📊 检测结果详细信息", expanded=True):
                        # 显示总检测数量
                        st.markdown(f"### 检测统计")
                        st.markdown(f"**总检测对象数**: {len(detections)}")
                        
                        # 显示各类别统计
                        if class_counts:
                            st.markdown("**各类别分布**:")
                            for class_name, count in class_counts.items():
                                st.markdown(f"- {class_name}: {count}")
                        else:
                            st.info("未检测到任何对象")
                        
                        # 显示详细检测信息
                        if detections:
                            st.markdown("### 详细检测信息")
                            # 使用表格显示检测结果
                            detection_table = []
                            for detection in detections:
                                detection_table.append({
                                    "ID": detection['id'],
                                    "类别": detection['class'],
                                    "置信度": f"{detection['confidence']*100:.1f}%",
                                    "X坐标": f"{detection['coordinates']['x']:.1f}",
                                    "Y坐标": f"{detection['coordinates']['y']:.1f}",
                                    "宽度": f"{detection['coordinates']['width']:.1f}",
                                    "高度": f"{detection['coordinates']['height']:.1f}"
                                })
                            st.table(detection_table)
                    
                except Exception as ex:
                    st.error(f"处理图像时出错: {str(ex)}")
                    st.exception(ex)
    else:
        st.info("请上传一张图像以进行检测")


def infer_uploaded_video(conf, model, is_tracking=False, tracker_type=None):
    """
    对上传的视频执行推理，可选跟踪功能
    
    :param conf: YOLOv8模型的置信度
    :param model: 包含YOLOv8模型的实例
    :param is_tracking: 是否启用对象跟踪
    :param tracker_type: 使用的跟踪器类型
    :return: None
    """
    st.markdown("### 视频" + ("跟踪" if is_tracking else "检测"))
    st.markdown(f"请上传一个视频文件，然后点击'开始处理'按钮进行{(is_tracking and '跟踪' or '检测')}")
    
    source_video = st.file_uploader(
        label="选择视频文件",
        type=("mp4", "avi", "mov", "mkv"),
        help="支持的视频格式: MP4, AVI, MOV, MKV"
    )

    # 显示上传的视频预览
    if source_video:
        # 使用边框和阴影效果的容器
        st.markdown('<div style="border: 1px solid #e5e7eb; border-radius: 0.5rem; padding: 0.5rem; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);">', unsafe_allow_html=True)
        st.video(source_video)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 执行按钮
        if st.button("开始处理", use_container_width=True):
            with st.spinner(f"正在加载视频并准备{(is_tracking and '跟踪' or '检测')}..."):
                try:
                    # 创建临时文件保存视频
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    tfile.write(source_video.read())
                    tfile.close()
                    
                    # 打开视频文件
                    vid_cap = cv2.VideoCapture(tfile.name)
                    
                    # 获取视频信息
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    total_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    st.info(f"视频信息: {fps:.1f} FPS, 约{total_frames//int(fps)}秒")
                    st.warning("处理视频可能需要一些时间，请耐心等待...")
                    
                    # 创建用于显示结果的容器
                    st_frame = st.empty()
                    
                    # 处理每一帧
                    frame_count = 0
                    while vid_cap.isOpened():
                        success, image = vid_cap.read()
                        if success:
                            frame_count += 1
                            # 更新状态信息
                            if frame_count % 10 == 0:  # 每10帧更新一次进度
                                progress = frame_count / total_frames * 100
                                st_frame.text(f"处理进度: {progress:.1f}%")
                            
                            # 显示检测/跟踪结果
                            _display_detected_frames(
                                conf,
                                model,
                                st_frame,
                                image,
                                is_tracking,
                                tracker_type
                            )
                        else:
                            break
                    
                    # 释放资源
                    vid_cap.release()
                    st.success(f"视频{(is_tracking and '跟踪' or '检测')}完成!")
                    
                except Exception as e:
                    st.error(f"处理视频时出错: {str(e)}")
                    st.exception(e)
    else:
        st.info("请上传一个视频文件以进行处理")


def infer_uploaded_webcam(conf, model, is_tracking=False, tracker_type=None):
    """
    使用网络摄像头执行实时推理，可选跟踪功能
    
    :param conf: YOLOv8模型的置信度
    :param model: 包含YOLOv8模型的实例
    :param is_tracking: 是否启用对象跟踪
    :param tracker_type: 使用的跟踪器类型
    :return: None
    """
    st.markdown("### 摄像头实时" + ("跟踪" if is_tracking else "检测"))
    st.markdown(f"点击'开始'按钮启动摄像头，进行{(is_tracking and '实时跟踪' or '实时检测')}")
    
    # 创建两个列，用于放置开始和停止按钮
    col_start, col_stop = st.columns([1, 1], gap="small")
    
    # 使用session_state来管理摄像头状态
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False
    
    with col_start:
        if st.button("开始", use_container_width=True, type="primary"):
            st.session_state.camera_active = True
    
    with col_stop:
        if st.button("停止", use_container_width=True, type="secondary"):
            st.session_state.camera_active = False
    
    # 显示摄像头状态信息
    status_text = "已启动" if st.session_state.camera_active else "已停止"
    status_color = "green" if st.session_state.camera_active else "red"
    st.markdown(f"**摄像头状态**: <span style='color:{status_color};font-weight:bold'>{status_text}</span>", unsafe_allow_html=True)
    
    # 当摄像头激活时，执行检测/跟踪
    if st.session_state.camera_active:
        try:
            # 打开摄像头
            vid_cap = cv2.VideoCapture(0)
            
            # 检查摄像头是否成功打开
            if not vid_cap.isOpened():
                st.error("无法访问摄像头，请检查设备连接或权限设置")
                st.session_state.camera_active = False
                return
            
            # 创建用于显示结果的容器
            st_frame = st.empty()
            
            # 显示一些提示信息
            st.info("按'停止'按钮结束摄像头检测")
            
            # 持续读取和处理摄像头帧
            frame_count = 0
            while st.session_state.camera_active:
                success, image = vid_cap.read()
                if success:
                    frame_count += 1
                    # 显示检测/跟踪结果
                    _display_detected_frames(
                        conf,
                        model,
                        st_frame,
                        image,
                        is_tracking,
                        tracker_type
                    )
                else:
                    st.error("无法读取摄像头帧")
                    break
            
            # 释放摄像头资源
            vid_cap.release()
            st_frame.empty()
            st.success("摄像头检测已停止")
            
        except Exception as e:
            st.error(f"处理摄像头时出错: {str(e)}")
            st.exception(e)
            # 确保摄像头被释放
            if 'vid_cap' in locals():
                vid_cap.release()
            st.session_state.camera_active = False
    else:
        st.info("点击'开始'按钮启动摄像头")
