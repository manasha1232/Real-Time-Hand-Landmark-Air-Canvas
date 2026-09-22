"""
Generate Demo Synthetic Video & Frames for Real-Time Hand Landmark Air Canvas.
Creates synthetic webcam test frames showing hands moving over an air canvas interface.
"""

import os
import cv2
import numpy as np
import math

def create_synthetic_hand_canvas_demo(output_dir="output", num_frames=90):
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, "demo_hand_canvas.mp4")
    image_path = os.path.join(output_dir, "demo_hand_frame.jpg")

    width, height = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))

    # Pre-define trajectory (drawing a star & spiral pattern)
    points = []
    center_x, center_y = 640, 400
    for i in range(num_frames):
        t = i * 0.1
        r = 150 + 50 * math.sin(5 * t)
        x = int(center_x + r * math.cos(t))
        y = int(center_y + r * math.sin(t))
        points.append((x, y))

    print(f"[INFO] Generating {num_frames} synthetic hand air canvas frames...")

    canvas_accum = np.ones((height, width, 3), dtype=np.uint8) * 255

    for frame_idx, (px, py) in enumerate(points):
        # Create dark studio background frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Background subtle gradient
        for y in range(0, height, 10):
            val = int(20 + 20 * (y / height))
            frame[y:y+10, :] = (val, val, val + 10)

        # Draw HUD Color Bar at top
        cv2.rectangle(frame, (0, 0), (width, 80), (40, 40, 40), -1)
        colors = [
            ("CLEAR", (255, 255, 255), (20, 10, 140, 70)),
            ("BLUE", (255, 0, 0), (160, 10, 280, 70)),
            ("GREEN", (0, 255, 0), (300, 10, 420, 70)),
            ("RED", (0, 0, 255), (440, 10, 560, 70)),
            ("YELLOW", (0, 255, 255), (580, 10, 700, 70)),
            ("PURPLE", (255, 0, 255), (720, 10, 840, 70)),
            ("ERASER", (128, 128, 128), (860, 10, 980, 70)),
        ]
        for name, col, rect in colors:
            cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), col, -1)
            cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (255, 255, 255), 2)
            cv2.putText(frame, name, (rect[0] + 10, rect[1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Select color based on frame progress
        if frame_idx < 30:
            current_color = (255, 0, 0) # Blue
        elif frame_idx < 60:
            current_color = (0, 255, 0) # Green
        else:
            current_color = (0, 0, 255) # Red

        # Draw accumulated brush stroke on canvas accum
        if frame_idx > 0:
            prev_pt = points[frame_idx - 1]
            cv2.line(canvas_accum, prev_pt, (px, py), current_color, 8)
            cv2.line(frame, prev_pt, (px, py), current_color, 8)

        # Draw synthetic hand representation (Index finger drawing)
        wrist = (px - 100, py + 180)
        index_tip = (px, py)
        index_pip = (px - 30, py + 60)
        index_mcp = (px - 50, py + 100)
        thumb_tip = (px - 80, py + 80)
        middle_tip = (px - 20, py + 120)

        # Draw hand skeleton lines
        hand_pts = [wrist, index_mcp, index_pip, index_tip]
        for i in range(len(hand_pts) - 1):
            cv2.line(frame, hand_pts[i], hand_pts[i+1], (0, 255, 255), 3)
        cv2.line(frame, index_mcp, thumb_tip, (0, 255, 255), 3)
        cv2.line(frame, index_mcp, middle_tip, (0, 255, 255), 3)

        # Draw hand landmark points
        for pt in [wrist, index_mcp, index_pip, thumb_tip, middle_tip]:
            cv2.circle(frame, pt, 8, (0, 165, 255), -1)
        # Highlight index tip with distinctive cyan color marker for detection
        cv2.circle(frame, index_tip, 12, (255, 255, 0), -1)
        cv2.circle(frame, index_tip, 16, (255, 255, 255), 2)

        # Overlay canvas drawing onto frame cleanly using array arithmetic
        mask = cv2.cvtColor(canvas_accum, cv2.COLOR_BGR2GRAY) < 250
        frame[mask] = (frame[mask].astype(np.float32) * 0.3 + canvas_accum[mask].astype(np.float32) * 0.7).astype(np.uint8)

        # Status text
        cv2.putText(frame, f"Frame: {frame_idx+1}/{num_frames} | Mode: DRAWING | Color: BGR{current_color}",
                    (20, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        out.write(frame)

        if frame_idx == num_frames - 1:
            cv2.imwrite(image_path, frame)

    out.release()
    print(f"[SUCCESS] Synthetic hand canvas video saved to: {video_path}")
    print(f"[SUCCESS] Sample test frame saved to: {image_path}")

if __name__ == "__main__":
    create_synthetic_hand_canvas_demo()
