#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   @File Name:     config.py
   @Author:        Luyao.zhang
   @Date:          2023/5/16
   @Description: configuration file
-------------------------------------------------
"""
from pathlib import Path
import sys

# Get the absolute path of the current file
file_path = Path(__file__).resolve()

# Get the parent directory of the current file
root_path = file_path.parent

# Add the root path to the sys.path list if it is not already there
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Get the relative path of the root directory with respect to the current working directory
# 修改为动态获取当前文件所在目录作为根目录
ROOT = root_path  # 使用当前文件的父目录作为根目录
# Source
SOURCES_LIST = ["Image", "Video", "Webcam"]


# DL model config
DETECTION_MODEL_DIR = ROOT / 'weights' / 'detection'
YOLOv8n = DETECTION_MODEL_DIR / "yolov8n.pt"
YOLOv8s = DETECTION_MODEL_DIR / "yolov8s.pt"
YOLOv8m = DETECTION_MODEL_DIR / "yolov8m.pt"
YOLOv8l = DETECTION_MODEL_DIR / "yolov8l.pt"
best = DETECTION_MODEL_DIR / "best.pt"
DETECTION_MODEL_LIST = [
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolov8l.pt",
    "yolov8x.pt",
    "best.pt"]

# Tracker config
TRACKER_TYPE_LIST = ["bytetrack", "botsort"]
DEFAULT_TRACKER = "bytetrack"
