import os
import tempfile
from collections import defaultdict
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from PIL import Image

# 1. ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="Pothole Detection & Tracking Dashboard",
    page_icon="🛣️",
    layout="wide"
)

st.title("🛣️ ระบบตรวจจับและประเมินระดับความเสียหายของหลุมบนผิวทาง")
st.markdown("ระบบวิเคราะห์สภาพถนนแบบเรียลไทม์ (YOLOv8s vs YOLO11s + ByteTrack)")
st.markdown("---")

# 2. เมนูตั้งค่าด้านข้าง (Sidebar)
st.sidebar.header("⚙️ การตั้งค่าระบบ")

model_choice = st.sidebar.selectbox(
    "เลือกโมเดลที่ต้องการใช้งาน: ",
    ("YOLOv8s (โมเดลหลัก - แนะนำ)", "YOLO11s"),
    index=0
)

# แมปพาทโมเดลอัตโนมัติ (รองรับทั้งชื่อ yolo8s.pt และ yolov8s.pt)
if "YOLOv8s" in model_choice:
    model_path = "models/yolo8s.pt" if os.path.exists("models/yolo8s.pt") else "models/yolov8s.pt"
else:
    model_path = "models/yolo11s.pt"

conf_threshold = st.sidebar.slider(
    "ค่าความเชื่อมั่นขั้นต่ำ (Confidence Threshold):",
    min_value=0.10,
    max_value=1.00,
    value=0.45,
    step=0.05
)

line_position_ratio = st.sidebar.slider(
    "ตำแหน่งเส้นตรวจจับสีเหลือง (ระดับความสูง):",
    min_value=0.40,
    max_value=0.85,
    value=0.70,
    step=0.05
)

source_radio = st.sidebar.radio(
    "เลือกแหล่งที่มาของข้อมูล:",
    ("วิดีโอตัวอย่าง (vdotest2.mp4)", "อัปโหลดวิดีโอใหม่", "อัปโหลดภาพนิ่ง"),
    index=0
)

@st.cache_resource
def load_yolo_model(path):
    return YOLO(path)

try:
    model = load_yolo_model(model_path)
    st.sidebar.success(f"โหลดโมเดล: `{os.path.basename(model_path)}` เรียบร้อย")
except Exception as e:
    st.sidebar.error(f"ไม่พบไฟล์โมเดล: {e}")
    st.stop()

# 3. จัดการข้อมูลนำเข้า
video_path = None
uploaded_image = None

if source_radio == "วิดีโอตัวอย่าง (vdotest2.mp4)":
    sample_path = "videos/vdotest2.mp4"
    if os.path.exists(sample_path):
        video_path = sample_path
    else:
        st.error(f"ไม่พบไฟล์วิดีโอที่ `{sample_path}` กรุณาตรวจสอบโฟลเดอร์ videos")

elif source_radio == "อัปโหลดวิดีโอใหม่":
    uploaded_video = st.sidebar.file_uploader("เลือกไฟล์วิดีโอ (.mp4, .avi, .mov)", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        video_path = tfile.name

elif source_radio == "อัปโหลดภาพนิ่ง":
    uploaded_image = st.sidebar.file_uploader("เลือกไฟล์ภาพ (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])

# 4. กรณีประมวลผลภาพนิ่ง
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    h, w, _ = img_cv.shape

    results = model.predict(img_cv, conf=conf_threshold, iou=0.30, verbose=False)
    
    minor_cnt = 0
    severe_cnt = 0
    detected_potholes_img = []

    if results[0].boxes is not None:
        raw_boxes = results[0].boxes.xyxy.cpu().numpy()
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        confidences = results[0].boxes.conf.cpu().numpy()

        for idx, (box, cls_id, conf) in enumerate(zip(raw_boxes, class_ids, confidences), start=1):
            x1, y1, x2, y2 = map(int, box)
            area = ((x2 - x1) / w) * ((y2 - y1) / h)
            is_severe = (cls_id == 1) or (area >= 0.020)

            if is_severe:
                severe_cnt += 1
                color = (0, 0, 255)
                name = "Severe Pothole"
            else:
                minor_cnt += 1
                color = (0, 255, 0)
                name = "Minor Pothole"

            # ครอปเก็บภาพหลุม
            c_y1, c_y2 = max(0, y1), min(h, y2)
            c_x1, c_x2 = max(0, x1), min(w, x2)
            if c_y2 > c_y1 and c_x2 > c_x1:
                crop = img_cv[c_y1:c_y2, c_x1:c_x2].copy()
                detected_potholes_img.append({
                    "id": idx,
                    "type": name,
                    "is_severe": is_severe,
                    "conf": float(conf),
                    "img": crop
                })

            cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
            label = f"{name} ({conf:.2f})"
            cv2.putText(img_cv, label, (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # กล่อง HUD บนภาพ
    cv2.rectangle(img_cv, (25, 25), (370, 135), (0, 0, 0), -1)
    cv2.putText(img_cv, f"Minor Potholes:  {minor_cnt}", (40, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.putText(img_cv, f"Severe Potholes: {severe_cnt}", (40, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    st.subheader("🖼️ ผลการวิเคราะห์ภาพถ่าย")
    if h > w:
        _, center_col, _ = st.columns([1.2, 1.8, 1.2])
        with center_col:
            st.image(img_cv, channels="BGR", use_container_width=True)
    else:
        st.image(img_cv, channels="BGR", use_container_width=True)

    # แสดงผลสรุปสำหรับภาพนิ่ง
    st.markdown("---")
    st.subheader("📊 สรุปผลการตรวจจับจากภาพถ่าย")
    c1, c2, c3 = st.columns(3)
    c1.metric("จำนวนหลุมทั้งหมด", f"{minor_cnt + severe_cnt} จุด")
    c2.metric("🟢 Minor Potholes", f"{minor_cnt} จุด")
    c3.metric("🔴 Severe Potholes", f"{severe_cnt} จุด")

# 5. กรณีประมวลผลวิดีโอ
elif video_path is not None:
    st.subheader("📹 การตรวจจับและนับสะสมบนวิดีโอ")
    
    col_btn1, col_btn2, _ = st.columns([1.5, 1.5, 7])
    with col_btn1:
        start_button = st.button("▶️ เริ่มประมวลผลวิดีโอ", use_container_width=True)
    with col_btn2:
        stop_button = st.button("⏹️ หยุดการทำงาน", use_container_width=True)

    # ตรวจสอบขนาดวิดีโอเพื่อปรับ Layout
    cap_temp = cv2.VideoCapture(video_path)
    vid_w = int(cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap_temp.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_temp.release()

    # ถ้าเป็นแนวตั้ง (9:16) บีบให้อยู่กึ่งกลาง ไม่ให้ยาวล้นจอ
    if vid_h > vid_w:
        _, center_col, _ = st.columns([1.2, 1.8, 1.2])
        with center_col:
            frame_window = st.empty()
    else:
        frame_window = st.empty()

    if start_button:
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        line_y = int(height * line_position_ratio)

        # ค่าจาก Jupyter
        SMOOTH_FACTOR = 0.60
        SEVERE_AREA_THRESHOLD = 0.020

        smoothed_boxes = {}
        track_max_area = defaultdict(float)
        counted_track_ids = set()

        # ตัวแปรเก็บภาพหลุมและยอดรวม
        detected_potholes = []
        total_minor = 0
        total_severe = 0

        while cap.isOpened():
            if stop_button:
                break

            ret, frame = cap.read()
            if not ret:
                break

            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=conf_threshold,
                iou=0.30,
                imgsz=640,
                verbose=False
            )

            # วาดเส้น Counting Line
            cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 255), 3)
            cv2.putText(frame, "COUNTING LINE", (40, line_y - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if results[0].boxes is not None and results[0].boxes.id is not None:
                raw_boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                class_ids = results[0].boxes.cls.int().cpu().tolist()
                confidences = results[0].boxes.conf.cpu().numpy()

                for box, track_id, cls_id, conf in zip(raw_boxes, track_ids, class_ids, confidences):
                    if track_id in smoothed_boxes:
                        box = smoothed_boxes[track_id] * SMOOTH_FACTOR + box * (1.0 - SMOOTH_FACTOR)
                    smoothed_boxes[track_id] = box

                    x1, y1, x2, y2 = map(int, box)

                    area = ((x2 - x1) / width) * ((y2 - y1) / height)
                    if area > track_max_area[track_id]:
                        track_max_area[track_id] = area

                    is_severe = (cls_id == 1) or (track_max_area[track_id] >= SEVERE_AREA_THRESHOLD)
                    class_name = "Severe Pothole" if is_severe else "Minor Pothole"
                    color = (0, 0, 255) if is_severe else (0, 255, 0)

                    # ตัดผ่านเส้น y1 <= line_y <= y2
                    if y1 <= line_y <= y2 and track_id not in counted_track_ids:
                        counted_track_ids.add(track_id)
                        if is_severe:
                            total_severe += 1
                        else:
                            total_minor += 1

                        # ครอปบันทึกภาพของหลุมจุดนี้เก็บไว้
                        crop_y1, crop_y2 = max(0, y1), min(height, y2)
                        crop_x1, crop_x2 = max(0, x1), min(width, x2)
                        if crop_y2 > crop_y1 and crop_x2 > crop_x1:
                            crop_patch = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
                            detected_potholes.append({
                                "id": track_id,
                                "type": class_name,
                                "is_severe": is_severe,
                                "conf": float(conf),
                                "img": crop_patch
                            })

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"#{track_id} {class_name} ({conf:.2f})"
                    cv2.putText(frame, label, (x1, max(22, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            # กล่อง HUD มุมซ้ายบนในวิดีโอ
            cv2.rectangle(frame, (25, 25), (370, 135), (0, 0, 0), -1)
            cv2.putText(frame, f"Minor Potholes:  {total_minor}", (40, 65), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            cv2.putText(frame, f"Severe Potholes: {total_severe}", (40, 105), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            # แสดงวิดีโอ
            frame_window.image(frame, channels="BGR", use_container_width=True)

        cap.release()

        # =========================================================
        # 6. แสดงผลสรุปสถิติและรูปภาพทั้งหมด (แสดงเมื่อนับเสร็จเท่านั้น)
        # =========================================================
        st.markdown("---")
        st.success("🎉 ประมวลผลวิดีโอเสร็จสิ้นเรียบร้อยแล้ว!")

        # 6.1 การ์ดสรุปตัวเลขสถิติรวม
        st.subheader("📊 สรุปผลการสำรวจและประเมินความเสียหายบนผิวทาง")
        stat_c1, stat_c2, stat_c3 = st.columns(3)
        total_potholes = total_minor + total_severe
        stat_c1.metric("จำนวนหลุมทั้งหมดที่ตรวจพบ", f"{total_potholes} จุด")
        stat_c2.metric("🟢 Minor Potholes (หลุมขนาดเล็ก)", f"{total_minor} จุด")
        stat_c3.metric("🔴 Severe Potholes (หลุมขนาดใหญ่/รุนแรง)", f"{total_severe} จุด")

        # 6.2 แสดงแกลเลอรีภาพหลุมที่ระบบแคปไว้
        if detected_potholes:
            st.markdown("### 📸 แกลเลอรีภาพหลุมที่บันทึกได้บนเส้นทาง (Pothole Log)")

            tab_all, tab_severe, tab_minor = st.tabs([
                f"ทั้งหมด ({len(detected_potholes)} จุด)",
                f"🔴 Severe ({total_severe} จุด)",
                f"🟢 Minor ({total_minor} จุด)"
            ])

            # ฟังก์ชันช่วยจัดเรียงแสดงภาพเป็นแถวละ 4 รูป
            def show_pothole_grid(potholes_subset):
                if not potholes_subset:
                    st.info("ไม่มีรายการหลุมในหมวดหมู่นี้")
                    return
                cols_per_row = 4
                for i in range(0, len(potholes_subset), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for c_idx, item in enumerate(potholes_subset[i : i + cols_per_row]):
                        with cols[c_idx]:
                            cap_text = f"#{item['id']} {item['type']} (Conf: {item['conf']:.2f})"
                            st.image(item["img"], caption=cap_text, channels="BGR", use_container_width=True)

            with tab_all:
                show_pothole_grid(detected_potholes)

            with tab_severe:
                show_pothole_grid([p for p in detected_potholes if p["is_severe"]])

            with tab_minor:
                show_pothole_grid([p for p in detected_potholes if not p["is_severe"]])
        else:
            st.info("ไม่พบหลุมที่เคลื่อนที่ผ่านเส้นตรวจจับ")