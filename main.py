import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import time
from datetime import datetime

FONT_PATHS = ["C:/Windows/Fonts/msjh.ttf", "C:/Windows/Fonts/mingliu.ttc", "msjh.ttf"] # You might need to change this to a font available on your system
FONT_SIZE = 20

# --- 參數設定 ---
CAP_WIDTH = 1280
CAP_HEIGHT = 720
BOX_COLOR = (255, 0, 255) # 亮粉色
COMMAND_ZONES = [
    (50, 50, 200, 100, "拍照 (Take Photo)"),
    (1030, 570, 200, 100, "播放影片 (Play Video)"),
    (50, 570, 200, 100, "結束程式 (Exit)"),
]
TRIGGER_THRESHOLD = 30
ACCUMULATION_RATE = 1
DECAY_RATE = 2
GET_READY_SECONDS = 3    # 按下按鍵後的準備時間
CALIBRATION_SECONDS = 3  # 正式校準時間
VAR_THRESHOLD = 75
CAMERA_WAIT_TIMEOUT = 10 # 等待攝影機啟動的最長時間（秒）

# --- 繪圖函式 ---
def draw_ui(frame, zones, accumulators=None, threshold=None):
    # 偵錯：重新排序繪圖順序，確保框線總是可見
    for i, (x, y, w, h, name) in enumerate(zones):
        # 1. 如果有提供進度，先畫進度條
        if accumulators is not None and threshold is not None:
            progress = min(accumulators[i] / threshold, 1.0)
            if progress > 0:
                cv2.rectangle(frame, (x, y), (x + int(w * progress), y + h), BOX_COLOR, -1)

        # 2. 接著畫框線
        cv2.rectangle(frame, (x, y), (x+w, y+h), BOX_COLOR, 3)
        
        # 3. 最後畫文字，確保文字在最上層
        # 使用 put_chinese_text 函式來顯示中文
        frame = put_chinese_text(frame, name, (x + 10, y + h - 15 - FONT_SIZE // 2), FONT_PATHS, FONT_SIZE, (255, 255, 255))
    return frame

def put_chinese_text(frame, text, position, font_paths, font_size, color):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    font = None
    for path in font_paths:
        try:
            font = ImageFont.truetype(path, font_size)
            break
        except IOError:
            continue

    if font is None:
        print(f"錯誤：找不到任何字體檔案於 {font_paths}，請確認路徑是否正確。將使用預設字體。")
        font = ImageFont.load_default()
    
    draw.text(position, text, font=font, fill=(color[2], color[1], color[0])) # OpenCV BGR to PIL RGB
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# --- 攝影機選擇函式 ---
def select_camera():
    print("正在偵測可用的攝影機...")
    available_cameras = []
    for i in range(5): # 嘗試檢查 0-4 的索引
        cap_test = cv2.VideoCapture(i)
        if cap_test.isOpened():
            ret, _ = cap_test.read()
            if ret:
                available_cameras.append(i)
            cap_test.release()
    
    if not available_cameras:
        print("錯誤：找不到任何可用的攝影機。")
        return None
    
    if len(available_cameras) == 1:
        print(f"自動選擇唯一的攝影機: {available_cameras[0]}")
        return available_cameras[0]

    print("請選擇要使用的攝影機：")
    for cam_idx in available_cameras:
        print(f"  - 輸入 {cam_idx} 選擇攝影機 {cam_idx}")
    
    while True:
        try:
            choice = int(input("請輸入攝影機編號: "))
            if choice in available_cameras:
                return choice
            else:
                print("無效的選擇，請重新輸入。")
        except ValueError:
            print("請輸入數字。")

# --- 程式主體 ---
def main():
    global COMMAND_ZONES
    # --- 新增：攝影機選擇 ---
    camera_index = select_camera()
    if camera_index is None:
        return

    # --- 修改：耐心等待攝影機啟動 ---
    print(f"正在啟動攝影機 {camera_index}，這可能需要幾秒鐘...")
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    
    start_time = time.time()
    camera_ready = False
    while time.time() - start_time < CAMERA_WAIT_TIMEOUT:
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                camera_ready = True
                break
        # 短暫延遲，避免 CPU 占用過高
        time.sleep(0.1)

    if not camera_ready:
        print(f"錯誤：在 {CAMERA_WAIT_TIMEOUT} 秒內無法從攝影機讀取畫面。")
        print("請檢查攝影機是否被其他程式占用，或重新插拔攝影機。")
        cap.release()
        return

    print("攝影機已成功啟動！")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
    
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=VAR_THRESHOLD, detectShadows=False)

    # 階段一：等待使用者按下按鍵
    window_name = "Hand Gesture Interface"
    cv2.namedWindow(window_name)

    while True:
        ret, frame = cap.read()
        if not ret: continue
        
        frame = cv2.flip(frame, 1)
        
        # 顯示操作提示
        cv2.putText(frame, "Press 's' to start calibration", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, "Press 'e' to edit layout", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, "Press 'q' to quit", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        frame = draw_ui(frame, COMMAND_ZONES)
        cv2.imshow(window_name, frame)
        
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            print("收到指令！準備開始校準...")
            break
        
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return

        elif key == ord('e'):
            COMMAND_ZONES = run_edit_mode(cap, COMMAND_ZONES, window_name, BOX_COLOR)
            # 繼續外層迴圈，等待 's' 或 'q'


    # 階段二：給使用者準備的倒數計時
    start_time = time.time()
    while time.time() - start_time < GET_READY_SECONDS:
        ret, frame = cap.read()
        if not ret: continue
        
        frame = cv2.flip(frame, 1)
        remaining_time = GET_READY_SECONDS - int(time.time() - start_time)
        cv2.putText(frame, f"Get Ready! Calibration starts in: {remaining_time}", 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        frame = draw_ui(frame, COMMAND_ZONES)
        cv2.imshow("Hand Gesture Interface", frame)
        cv2.waitKey(1)

    # 階段三：正式校準
    print(f"即將進行 {CALIBRATION_SECONDS} 秒背景校準，請保持畫面靜止...")
    start_time = time.time()
    while time.time() - start_time < CALIBRATION_SECONDS:
        ret, frame = cap.read()
        if not ret: continue
        
        frame = cv2.flip(frame, 1)
        remaining_time = CALIBRATION_SECONDS - int(time.time() - start_time)
        cv2.putText(frame, f"Calibrating... Keep Still: {remaining_time}", 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        frame = draw_ui(frame, COMMAND_ZONES)
        cv2.imshow("Hand Gesture Interface", frame)
        cv2.waitKey(1)
        
        blur_frame = cv2.GaussianBlur(frame, (5, 5), 0)
        backSub.apply(blur_frame)

    print("校準完成，可以開始操作！")
    zone_accumulators = [0] * len(COMMAND_ZONES)
    feedback_timers = [0] * len(COMMAND_ZONES)

    # 階段四：主偵測迴圈
    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        blur_frame = cv2.GaussianBlur(frame, (5, 5), 0)
        fg_mask = backSub.apply(blur_frame)
        fg_mask = cv2.erode(fg_mask, None, iterations=2)
        fg_mask = cv2.dilate(fg_mask, None, iterations=2)

        for i, (x, y, w, h, name) in enumerate(COMMAND_ZONES):
            roi_mask = fg_mask[y:y+h, x:x+w]
            motion_detected = cv2.countNonZero(roi_mask)

            if motion_detected > (w * h * 0.1): zone_accumulators[i] += ACCUMULATION_RATE
            else: zone_accumulators[i] = max(0, zone_accumulators[i] - DECAY_RATE)

            if zone_accumulators[i] > TRIGGER_THRESHOLD:
                print(f"指令觸發: {name}")
                feedback_timers[i] = time.time()

                if name == "拍照 (Take Photo)":
                    output_dir = "photos"
                    os.makedirs(output_dir, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(output_dir, f"photo_{timestamp}.jpg")
                    cv2.imwrite(filename, frame)
                    print(f"照片已儲存至: {filename}")

                elif name == "播放影片 (Play Video)":
                    video_path = "高清版瑞克搖.mp4"
                    if not os.path.exists(video_path):
                        print(f"錯誤：影片檔案 '{video_path}' 不存在。")
                    else:
                        player = cv2.VideoCapture(video_path)
                        if not player.isOpened():
                            print(f"錯誤：無法開啟影片檔案 '{video_path}'。")
                        else:
                            print(f"正在播放影片: {video_path}")
                            # 隱藏主視窗，避免影片播放時干擾
                            cv2.destroyWindow("Hand Gesture Interface")
                            cv2.destroyWindow("Foreground Mask")

                            while True:
                                ret_video, frame_video = player.read()
                                if not ret_video:
                                    break
                                cv2.imshow("Video Player", frame_video)
                                if cv2.waitKey(25) & 0xFF == ord('q'):
                                    break
                            player.release()
                            cv2.destroyWindow("Video Player")
                            print("影片播放結束。")
                            # 重新顯示主視窗
                            cv2.namedWindow("Hand Gesture Interface")
                            cv2.namedWindow("Foreground Mask")

                if name == "結束程式 (Exit)":
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                
                zone_accumulators[i] = 0
        
        
        # --- 新增：處理視覺回饋 ---
        for i, start_time in enumerate(feedback_timers):
            if start_time > 0:
                elapsed_time = time.time() - start_time
                if elapsed_time < 5.0:  # 持續 5 秒
                    x, y, w, h, _ = COMMAND_ZONES[i]
                    
                    # 建立一個半透明的綠色疊加層
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), -1) # 綠色
                    alpha = 0.4  # 透明度
                    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
                else:
                    feedback_timers[i] = 0 # 重置計時器
        
        frame = draw_ui(frame, COMMAND_ZONES, zone_accumulators, TRIGGER_THRESHOLD)

        cv2.imshow("Hand Gesture Interface", frame)
        cv2.imshow("Foreground Mask", fg_mask) 

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    print("程式已結束。")

if __name__ == '__main__':
    main()