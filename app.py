from flask import Flask, render_template, Response, request, jsonify, send_from_directory
import pandas as pd
import os
import logging
import cv2
import joblib
import numpy as np
from ultralytics import YOLO, RTDETR
import yt_dlp
from collections import deque, defaultdict
from datetime import datetime
import pytz
import time

# 전역변수 선언
has_new_warning = False
current_warning_cam = None
warning_histories = {}  # track_id별 warning 이력
warning_actives = defaultdict(bool)  # track_id별 warning 활성화 상태
warning_times = {}  # track_id별 warning 시간
person_limits = {'video_feed_1': 1, 'video_feed_2': 1}  # 스트림별 인원 제한 기본값을 1로 설정
total_person_count = 0  # 전역 변수로 총 인원수 초기화
stream_states = {}
frame_skip = 1  # 초당 약 10프레임 처리 (30fps 기준)
warning_threshold = 20  # 7초 중 3초 이상 (30fps 기준, 3초 = 30프레임)
max_history_length = 70  # 7초 (30fps 기준, 7초 = 70프레임)

# 이미지 저장 디렉토리 설정
IMG_DATA_DIR = 'img_data'
os.makedirs(os.path.join(IMG_DATA_DIR, 'cam1'), exist_ok=True)
os.makedirs(os.path.join(IMG_DATA_DIR, 'cam2'), exist_ok=True)

def init_stream_state():
    return {
        'warning_histories': {},
        'warning_actives': {},
        'warning_times': {},
        'frame_count': 0,
        'current_count': 0,
        'yolo_model': YOLO('/home/siren/flask_yes/real_NoneML/models/pick_4cls_8x_mi_225e_best.pt'),
        'rtdetr_model': RTDETR('/home/siren/flask_yes/real_NoneML/models/PICK_4cls_RTDETR_L.pt'),
        'vote_histories': {},
        'vote_windows': {}
    }

def calculate_box_iou(box1, box2):
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = box1_area + box2_area - intersection
    
    return intersection / (union + 1e-7)

def generate_frames_common(stream_id, cap, cam_no):
    global has_new_warning, current_warning_cam

    if stream_id not in stream_states:
        stream_states[stream_id] = init_stream_state()
    
    state = stream_states[stream_id]
    warning_histories = state['warning_histories']
    warning_actives = state['warning_actives']
    warning_times = state['warning_times']
    frame_count = state['frame_count']
    yolo_model = state['yolo_model']
    rtdetr_model = state['rtdetr_model']
    vote_histories = state['vote_histories']
    vote_windows = state['vote_windows']

    skip_count = 0
    FRAME_SKIP = 3
    conf_threshold = 0.3
    iou_threshold = 0.45
    yolo_weight = 0.1
    rtdetr_weight = 0.9
    
    VOTE_WINDOW_SIZE = 5

    while True:
        try:
            ret, frame = cap.read()
            if not ret or frame is None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            skip_count += 1
            if skip_count % FRAME_SKIP != 0:
                continue

            frame = cv2.resize(frame, (400, 300))
            
            # YOLO와 RTDETR 모델 추론
            yolo_results = yolo_model(frame, verbose=False, conf=conf_threshold)[0]
            rtdetr_results = rtdetr_model(frame, verbose=False, conf=conf_threshold)[0]
            
            # YOLO 결과 처리
            yolo_detections = []
            if len(yolo_results.boxes) > 0:
                yolo_boxes = yolo_results.boxes.xyxy.cpu().numpy()
                yolo_classes = yolo_results.boxes.cls.cpu().numpy().astype(int)
                yolo_confs = yolo_results.boxes.conf.cpu().numpy()
                yolo_detections = [{'box': box, 'class': cls, 'conf': conf} 
                                 for box, cls, conf in zip(yolo_boxes, yolo_classes, yolo_confs)]
            
            # RTDETR 결과 처리
            rtdetr_detections = []
            if len(rtdetr_results.boxes) > 0:
                rtdetr_boxes = rtdetr_results.boxes.xyxy.cpu().numpy()
                rtdetr_classes = rtdetr_results.boxes.cls.cpu().numpy().astype(int)
                rtdetr_confs = rtdetr_results.boxes.conf.cpu().numpy()
                rtdetr_detections = [{'box': box, 'class': cls, 'conf': conf} 
                                   for box, cls, conf in zip(rtdetr_boxes, rtdetr_classes, rtdetr_confs)]
            
            # 결과 통합
            final_detections = []
            used_yolo = set()
            used_rtdetr = set()
            
            # 신뢰도 순으로 정렬
            yolo_detections.sort(key=lambda x: x['conf'], reverse=True)
            rtdetr_detections.sort(key=lambda x: x['conf'], reverse=True)
            
            # YOLO와 RTDETR 박스 매칭 및 통합
            for i, yolo_det in enumerate(yolo_detections):
                best_iou = 0
                best_rtdetr_idx = -1
                
                for j, rtdetr_det in enumerate(rtdetr_detections):
                    if j in used_rtdetr:
                        continue
                    
                    iou = calculate_box_iou(yolo_det['box'], rtdetr_det['box'])
                    if iou > best_iou:
                        best_iou = iou
                        best_rtdetr_idx = j
                
                # IoU가 threshold보다 높은 경우 두 결과 통합
                if best_iou > iou_threshold and best_rtdetr_idx not in used_rtdetr:
                    rtdetr_det = rtdetr_detections[best_rtdetr_idx]
                    
                    # 같은 클래스로 예측한 경우
                    if yolo_det['class'] == rtdetr_det['class']:
                        final_class = yolo_det['class']
                        final_conf = (yolo_det['conf'] * yolo_weight + rtdetr_det['conf'] * rtdetr_weight)
                    else:
                        # 가중치를 적용한 신뢰도 비교
                        yolo_weighted_conf = yolo_det['conf'] * yolo_weight
                        rtdetr_weighted_conf = rtdetr_det['conf'] * rtdetr_weight
                        
                        if yolo_weighted_conf > rtdetr_weighted_conf:
                            final_class = yolo_det['class']
                            final_conf = yolo_det['conf']
                        else:
                            final_class = rtdetr_det['class']
                            final_conf = rtdetr_det['conf']
                    
                    final_detections.append({
                        'box': yolo_det['box'],
                        'class': final_class,
                        'conf': final_conf
                    })
                    
                    used_yolo.add(i)
                    used_rtdetr.add(best_rtdetr_idx)
            
            # 매칭되지 않은 검출 결과 처리
            for i, det in enumerate(yolo_detections):
                if i not in used_yolo and det['conf'] * yolo_weight > conf_threshold:
                    det = det.copy()
                    det['conf'] = det['conf'] * yolo_weight
                    final_detections.append(det)
            
            for i, det in enumerate(rtdetr_detections):
                if i not in used_rtdetr and det['conf'] * rtdetr_weight > conf_threshold:
                    det = det.copy()
                    det['conf'] = det['conf'] * rtdetr_weight
                    final_detections.append(det)
            
            # 최종 결과 시각화 및 처리
            current_tracks = set()
            person_count = 0
            
            for detection in final_detections:
                box = detection['box']
                final_class = detection['class']
                final_conf = detection['conf']
                
                x1, y1, x2, y2 = map(int, box)
                track_id = len(current_tracks)
                current_tracks.add(track_id)
                
                if int(final_class) in [0, 1]:
                    person_count += 1
                
                if int(final_class) == 1:  # lying_down
                    label = "lying_down"
                    color = (0, 0, 255)
                    warning_detected = True
                else:
                    label = "person"
                    color = (0, 255, 0)
                    warning_detected = False
                
                # 보팅 윈도우 업데이트
                if track_id not in vote_windows:
                    vote_windows[track_id] = deque(maxlen=VOTE_WINDOW_SIZE)
                vote_windows[track_id].append(warning_detected)
                
                warning_status = sum(vote_windows[track_id]) / len(vote_windows[track_id]) >= 0.6
                
                if track_id not in warning_histories:
                    warning_histories[track_id] = deque(maxlen=max_history_length)
                warning_histories[track_id].append(1 if warning_status else 0)
                
                # 시각화
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                text = f"ID:{track_id} {label} ({final_conf:.2f})"
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Warning 처리
                if warning_detected:
                    if track_id not in warning_histories:
                        warning_histories[track_id] = deque(maxlen=max_history_length)
                    warning_histories[track_id].append(1)
                    
                    if len(warning_histories[track_id]) >= max_history_length:
                        fall_count = sum(warning_histories[track_id])
                        if fall_count >= warning_threshold and not warning_actives.get(track_id, False):
                            has_new_warning = True
                            current_warning_cam = stream_id
                            warning_actives[track_id] = True
                            warning_times[track_id] = datetime.now(pytz.timezone('Asia/Seoul'))
                            try:
                                log_warning(track_id, frame, cam_no)
                            except Exception as e:
                                print(f"Warning logging error: {str(e)}")
                else:
                    if track_id in warning_histories:
                        warning_histories[track_id].append(0)
            
            # 현재 인원 수 업데이트 및 표시
            state['current_count'] = person_count
            cv2.putText(frame, f'Person Count: {person_count}', (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 제한 인원 초과 경고
            limit = person_limits.get(stream_id, 1)
            if person_count > limit:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (400, 50), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
                message = f"DENSITY WARNING! ({person_count}/{limit})"
                cv2.putText(frame, message, (10, 35), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                   
        except Exception as e:
            print(f"Error in generate_frames: {str(e)}")
            continue

def log_warning(track_id, frame, cam_no):
    try:
        # 현재 시간 가져오기
        current_time = datetime.now(pytz.timezone('Asia/Seoul'))
        
        # 이미지 저장
        cam_folder = f'cam{cam_no}'
        image_filename = f"{current_time.strftime('%Y%m%d_%H%M%S')}_{track_id}.jpg"
        image_path = os.path.join(cam_folder, image_filename)
        cv2.imwrite(os.path.join(IMG_DATA_DIR, image_path), frame)
        
        # CSV 파일에 기록
        if not os.path.exists('csvdata/warnings.csv'):
            # csvdata 디렉토리가 없으면 생성
            os.makedirs(os.path.dirname('csvdata/warnings.csv'), exist_ok=True)
            df = pd.DataFrame(columns=['cam_no', 'timestamp', 'object_id', 'image_path'])
            df.to_csv('csvdata/warnings.csv', index=False)
        
        # 새로운 경고 추가
        new_warning = pd.DataFrame({
            'cam_no': [cam_no],
            'timestamp': [current_time.strftime("%Y-%m-%d %H:%M:%S")],
            'object_id': [track_id],
            'image_path': [image_path]  # 상대 경로로 저장
        })
        
        # 기존 CSV 파일에 추가
        df = pd.read_csv('csvdata/warnings.csv')
        df = pd.concat([df, new_warning], ignore_index=True)
        df.to_csv('csvdata/warnings.csv', index=False)
        
    except Exception as e:
        print(f"Error logging warning: {str(e)}")

app = Flask(__name__)

# CSV 파일 경로 설정
csv_file_path = 'csvdata/warnings.csv'
saver_csv_path = 'csvdata/saver.csv'

# CSV 파일이 존재하지 않을 경우에만 새로 생성
if not os.path.isfile(csv_file_path):
    df = pd.DataFrame(columns=['cam_no', 'timestamp', 'object_id', 'image_path'])
    df.to_csv(csv_file_path, index=False)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_warnings')
def get_warnings():
    try:
        # CSV 파일이 없으면 빈 리스트 반환
        if not os.path.exists(csv_file_path):
            print("Warning: CSV file does not exist")
            return jsonify([])
            
        # CSV 파일 읽기
        df = pd.read_csv(csv_file_path)
        print(f"Read {len(df)} rows from CSV")
        
        # 데이터가 비어있으면 빈 리스트 반환
        if df.empty:
            print("Warning: CSV file is empty")
            return jsonify([])
            
        # 데이터 타입 변환
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['object_id'] = df['object_id'].astype(int)
        df['cam_no'] = df['cam_no'].astype(int)
        
        # 카메라별로 그룹화하고 각 그룹 내에서 시간 기준 내림차순 정렬
        df = df.sort_values(['cam_no', 'timestamp'], ascending=[True, False])
        
        # 데이터를 딕셔너리 리스트로 변환
        warnings = []
        for _, row in df.iterrows():
            warning = {
                'timestamp': row['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'object_id': int(row['object_id']),
                'cam_no': int(row['cam_no']),
                'image_path': row['image_path']
            }
            warnings.append(warning)
            
        print(f"Returning {len(warnings)} warnings")
        return jsonify(warnings)
        
    except Exception as e:
        print(f"Error getting warnings: {str(e)}")
        return jsonify([])

@app.route('/delete_warnings', methods=['POST'])
def delete_warnings():
    try:
        if not os.path.exists(csv_file_path):
            return jsonify({'success': False, 'error': 'No warnings file exists'})
            
        selected_items = request.json.get('selected_items', [])
        if not selected_items:
            return jsonify({'success': False, 'error': 'No items selected'})
            
        df = pd.read_csv(csv_file_path)
        if df.empty:
            return jsonify({'success': False, 'error': 'No warnings to delete'})
            
        df['object_id'] = df['object_id'].astype(int)
        df['cam_no'] = df['cam_no'].astype(int)
        
        # 선택된 항목 삭제
        for item in selected_items:
            object_id = int(item['object_id'])
            timestamp = item['timestamp']
            cam_no = int(item['cam_no'])
            
            mask = ~((df['object_id'] == object_id) & 
                    (df['timestamp'] == timestamp) & 
                    (df['cam_no'] == cam_no))
            df = df[mask]
        
        # 변경된 데이터 저장
        df.to_csv(csv_file_path, index=False)
        
        # 업데이트된 경고 목록 반환
        warnings = []
        for _, row in df.iterrows():
            warnings.append({
                'timestamp': row['timestamp'],
                'object_id': int(row['object_id']),
                'cam_no': int(row['cam_no']),
                'image_path': row['image_path']
            })
            
        return jsonify({'success': True, 'warnings': warnings})
        
    except Exception as e:
        print(f"Error deleting warnings: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/submit_rescue', methods=['POST'])
def submit_rescue():
    try:
        selected_items = request.json['selected_items']
        rescue_text = request.json['rescue_text']
        
        warnings_df = pd.read_csv(csv_file_path)
        warnings_df['object_id'] = warnings_df['object_id'].astype(int)
        warnings_df['cam_no'] = warnings_df['cam_no'].astype(int)
        
        selected_warnings = pd.DataFrame()
        for item in selected_items:
            object_id = int(item['object_id'])
            timestamp = item['timestamp']
            cam_no = int(item['cam_no'])
            selected_row = warnings_df[(warnings_df['object_id'] == object_id) & (warnings_df['timestamp'] == timestamp) & (warnings_df['cam_no'] == cam_no)]
            selected_warnings = pd.concat([selected_warnings, selected_row])

        # 구조 요청 텍스트 추가
        selected_warnings['rescue_text'] = rescue_text

        # saver.csv에 데이터 추가
        if os.path.exists(saver_csv_path):
            saver_df = pd.read_csv(saver_csv_path)
            saver_df = pd.concat([saver_df, selected_warnings], ignore_index=True)
        else:
            saver_df = selected_warnings
        saver_df.to_csv(saver_csv_path, index=False)

        # warnings.csv에서 선택된 데이터 삭제
        warnings_df = warnings_df[~((warnings_df['object_id'].isin(selected_warnings['object_id'])) & 
                                    (warnings_df['timestamp'].isin(selected_warnings['timestamp'])) & 
                                    (warnings_df['cam_no'].isin(selected_warnings['cam_no'])))]
        warnings_df.to_csv(csv_file_path, index=False)

        return jsonify({'success': True, 'warnings': warnings_df.to_dict('records')})
    except Exception as e:
        print(f"Error processing rescue request: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/set_person_limit', methods=['POST'])
def set_person_limit():
    global person_limits
    data = request.get_json()
    try:
        stream_id = data.get('stream_id')
        new_limit = int(data.get('limit'))
        if new_limit <= 0:
            return jsonify({'success': False, 'error': '인원 제한은 양수여야 합니다.'})
        person_limits[stream_id] = new_limit
        return jsonify({'success': True})
    except (TypeError, ValueError):
        return jsonify({'success': False, 'error': '올바른 숫자를 입력해주세요.'})

@app.route('/check_density', methods=['GET'])
def check_density():
    global person_limits, total_person_count
    # 각 스트림의 인원수 합산
    total_count = 0
    for state in stream_states.values():
        if hasattr(state, 'current_count'):
            total_count += state.get('current_count', 0)
    
    total_person_count = total_count  # 전역 변수 업데이트
    
    return jsonify({
        'current_count': total_person_count,
        'limits': person_limits,
        'is_over_limit': {stream_id: count > person_limits.get(stream_id, 1) for stream_id, count in stream_states.items() if hasattr(stream_states[stream_id], 'current_count')}
    })

@app.route('/check_new_warning')
def check_new_warning():
    global has_new_warning, current_warning_cam
    
    warning = False
    cam = None
    
    if has_new_warning and current_warning_cam:
        warning = True
        cam = current_warning_cam.replace('stream', '')  # stream1 -> 1
        
    response = {
        'warning': warning,
        'cam': cam
    }
    
    # 응답 후 상태 초기화
    has_new_warning = False
    current_warning_cam = None
    
    return jsonify(response)

@app.route('/video_feed_1')
def video_feed_1():
    stream_id = 'stream1'
    url = "https://www.youtube.com/watch?v=UhDQE0SMBGI"
    ydl_opts = {
        'format': 'best[height<=720]',
        'quiet': True,
        'no_warnings': True,
        'ignoreerrors': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        if info is None:
            print(f"Error: Could not extract info for {url}")
            return b''
        stream_url = info['url']
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print(f"Error: Could not open video stream for {stream_id}")
        return b''
    return Response(generate_frames_common(stream_id, cap, 1),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_2')
def video_feed_2():
    stream_id = 'stream2'
    url = "https://www.youtube.com/watch?v=qre2Z0Lvhj8"
    ydl_opts = {
        'format': 'best[height<=720]',
        'quiet': True,
        'no_warnings': True,
        'ignoreerrors': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        if info is None:
            print(f"Error: Could not extract info for {url}")
            return b''
        stream_url = info['url']
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print(f"Error: Could not open video stream for {stream_id}")
        return b''
    return Response(generate_frames_common(stream_id, cap, 2),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/img_data/<path:filename>')
def serve_image(filename):
    return send_from_directory('img_data', filename)

if __name__ == '__main__':
    stream_states.clear()  # 전역 변수 초기화
    app.run(debug=True, host='0.0.0.0', threaded=True)