#!/usr/bin/env python3
"""
Realtime gaze demo for ETH-XGaze, adapted from the image-based demo.

Features
- Webcam/Video capture
- dlib CNN *or* HOG face detection (configurable)
- 68 or 5-landmark predictor (configurable)
- Landmark smoothing (exponential moving average)
- Optical flow landmark tracking between detections (PyrLK)
- Periodic re-detection to correct drift
- Head pose + face normalization identical to demo
- Gaze inference with ETH-XGaze gaze_network
- CPU/GPU toggle

Usage (examples)
- python realtime_gaze.py --cnn --predictor modules/shape_predictor_68_face_landmarks.dat \
    --mmod modules/mmod_human_face_detector.dat --ckpt ckpt/epoch_24_ckpt.pth.tar
- python realtime_gaze.py --video path/to/video.mp4

Notes
- If camera calibration XML is missing, uses sensible defaults.
- Press 'q' to quit.
"""

import os
import time
import math
import argparse
from collections import deque

import cv2
import dlib
import numpy as np
import torch
from torchvision import transforms
from imutils import face_utils

# ====== ETH-XGaze model import ======
try:
    from model import gaze_network  # from ETH-XGaze repo
except Exception as e:
    raise RuntimeError("Could not import gaze_network from ETH-XGaze repo. Make sure this file lives inside the repo root.") from e

# ---------- Transforms (same as demo) ----------
trans = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---------- Utils ----------
class EMA:
    """Exponential Moving Average for smoothing points/angles."""
    def __init__(self, alpha=0.6):
        self.alpha = alpha
        self.state = None

    def reset(self):
        self.state = None

    def update(self, x):
        x = np.asarray(x, dtype=np.float32)
        if self.state is None:
            self.state = x
        else:
            self.state = self.alpha * x + (1.0 - self.alpha) * self.state
        return self.state


def estimate_head_pose(landmarks, face_model, camera, distortion, iterate=True):
    ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, flags=cv2.SOLVEPNP_EPNP)
    if iterate:
        ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, rvec, tvec, True)
    return rvec, tvec


def draw_gaze(image_in, pitchyaw, thickness=2, color=(0, 0, 255)):
    img = image_in
    h, w = img.shape[:2]
    length = min(h, w) / 2.0
    pos = (int(w / 2.0), int(h / 2.0))
    if img.ndim == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    dx = -length * math.sin(float(pitchyaw[1])) * math.cos(float(pitchyaw[0]))
    dy = -length * math.sin(float(pitchyaw[0]))
    cv2.arrowedLine(img, pos, (int(round(pos[0] + dx)), int(round(pos[1] + dy))), color, thickness, cv2.LINE_AA, tipLength=0.2)
    return img


def normalize_face(img, face_model, landmarks, hr, ht, cam):
    focal_norm = 960
    distance_norm = 600
    roiSize = (224, 224)

    ht = ht.reshape((3, 1))
    hR = cv2.Rodrigues(hr)[0]
    Fc = (hR @ face_model.T) + ht
    two_eye_center = np.mean(Fc[:, 0:4], axis=1).reshape((3, 1))
    nose_center = np.mean(Fc[:, 4:6], axis=1).reshape((3, 1))
    face_center = np.mean(np.concatenate((two_eye_center, nose_center), axis=1), axis=1).reshape((3, 1))

    distance = np.linalg.norm(face_center)
    z_scale = distance_norm / distance
    cam_norm = np.array([[focal_norm, 0, roiSize[0] / 2], [0, focal_norm, roiSize[1] / 2], [0, 0, 1.0]])
    S = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, z_scale]])

    hRx = hR[:, 0]
    forward = (face_center / distance).reshape(3)
    down = np.cross(forward, hRx); down /= np.linalg.norm(down)
    right = np.cross(down, forward); right /= np.linalg.norm(right)
    R = np.c_[right, down, forward].T

    W = cam_norm @ S @ R @ np.linalg.inv(cam)
    img_warped = cv2.warpPerspective(img, W, roiSize)

    hR_norm = R @ hR
    hr_norm = cv2.Rodrigues(hR_norm)[0]

    num_point = landmarks.shape[0]
    landmarks_warped = cv2.perspectiveTransform(landmarks, W).reshape(num_point, 2)

    return img_warped, landmarks_warped, hr_norm


def load_camera_params(xml_path, default_w=224, default_h=224):
    if os.path.isfile(xml_path):
        fs = cv2.FileStorage(xml_path, cv2.FILE_STORAGE_READ)
        camera_matrix = fs.getNode('Camera_Matrix').mat()
        camera_distortion = fs.getNode('Distortion_Coefficients').mat()
        if camera_matrix is None or camera_distortion is None:
            raise RuntimeError(f"Camera file {xml_path} missing required nodes.")
        return camera_matrix.astype(np.float32), camera_distortion.astype(np.float32)
    # Fallback defaults
    cx, cy = default_w / 2, default_h / 2
    camera_matrix = np.array([[1000, 0, cx], [0, 1000, cy], [0, 0, 1]], dtype=np.float32)
    camera_distortion = np.zeros((5, 1), dtype=np.float32)
    return camera_matrix, camera_distortion


def to_rect_dlib(rect):
    return dlib.rectangle(int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))


def detect_faces_dlib(frame_rgb, detector, use_cnn, upsample=1):
    if use_cnn:
        results = detector(frame_rgb, upsample)
        rects = [d.rect for d in results]
    else:
        rects = detector(frame_rgb, upsample)
    return rects


def landmarks_from_predictor(img_bgr, rect, predictor):
    shape = predictor(img_bgr, rect)
    shape = face_utils.shape_to_np(shape)
    return shape  # (68,2) or (5,2)


def track_landmarks_optical_flow(prev_gray, gray, prev_pts):
    lk_params = dict(winSize=(21, 21), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    next_pts, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)
    good_new = next_pts[st == 1]
    good_old = prev_pts[st == 1]
    return good_new.reshape(-1, 2), good_old.reshape(-1, 2), st


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', type=str, default='', help='Path to video file; if empty, use webcam 0')
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--cnn', action='store_true', help='Use dlib CNN face detector (requires mmod file)')
    ap.add_argument('--mmod', type=str, default='modules/mmod_human_face_detector.dat')
    ap.add_argument('--predictor', type=str, default='modules/shape_predictor_68_face_landmarks.dat')
    ap.add_argument('--ckpt', type=str, default='ckpt/epoch_24_ckpt.pth.tar')
    ap.add_argument('--face_model', type=str, default='face_model.txt')
    ap.add_argument('--camxml', type=str, default='', help='Optional camera calibration XML; defaults applied if missing')
    ap.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    ap.add_argument('--detect-interval', type=int, default=10, help='Run face detection every N frames')
    ap.add_argument('--scale', type=float, default=0.75, help='Scale factor for faster detection (0.5-1.0)')
    ap.add_argument('--ema-alpha', type=float, default=0.6, help='EMA smoothing factor (0-1)')
    args = ap.parse_args()

    # Video source
    cap = cv2.VideoCapture(0 if args.video == '' else args.video)
    if args.video == '':
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        raise RuntimeError('Could not open video source')

    # Face detector
    if args.cnn:
        if not os.path.isfile(args.mmod):
            raise FileNotFoundError(f"CNN face detector file not found: {args.mmod}")
        face_detector = dlib.cnn_face_detection_model_v1(args.mmod)
    else:
        face_detector = dlib.get_frontal_face_detector()

    # Landmark predictor
    if not os.path.isfile(args.predictor):
        raise FileNotFoundError(f"Predictor file not found: {args.predictor}")
    predictor = dlib.shape_predictor(args.predictor)

    # Camera params
    camera_matrix, camera_distortion = load_camera_params(args.camxml)

    # Gaze model
    model = gaze_network()
    use_cuda = (args.device == 'cuda' and torch.cuda.is_available())
    if use_cuda:
        model.cuda()
    model.eval()

    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f"Pretrained gaze ckpt missing: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location='cuda' if use_cuda else 'cpu')
    model.load_state_dict(ckpt['model_state'], strict=True)

    # Face model + indices same as demo
    face_model_load = np.loadtxt(args.face_model)
    landmark_use = [20, 23, 26, 29, 15, 19]  # eye corners + nose corners
    face_model = face_model_load[landmark_use, :]
    facePts = face_model.reshape(6, 1, 3)

    # Landmark indices on 68-point predictor corresponding to above
    idx = np.array([36, 39, 42, 45, 31, 35])

    # Smoothing
    gaze_ema = EMA(alpha=args.ema_alpha)
    lmk_ema = EMA(alpha=args.ema_alpha)

    # Tracking state
    prev_gray = None
    tracked_pts = None  # (N,2)
    last_detect_frame = -999
    frame_id = 0

    fps_hist = deque(maxlen=30)

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        orig = frame.copy()

        # Optionally downscale for detection
        run_detection = (frame_id - last_detect_frame) >= args.detect_interval or tracked_pts is None or tracked_pts.shape[0] < 6

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if run_detection:
            last_detect_frame = frame_id
            if 0 < args.scale < 1.0:
                small = cv2.resize(frame, None, fx=args.scale, fy=args.scale)
                small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                rects = detect_faces_dlib(small_rgb, face_detector, args.cnn, upsample=1)
                # rescale rects back
                rects = [dlib.rectangle(int(r.left()/args.scale), int(r.top()/args.scale), int(r.right()/args.scale), int(r.bottom()/args.scale)) for r in rects]
            else:
                rects = detect_faces_dlib(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), face_detector, args.cnn, upsample=1)

            if len(rects) == 0:
                cv2.putText(frame, 'No face', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                tracked_pts = None
            else:
                rect = rects[0]
                shape = landmarks_from_predictor(frame, rect, predictor)
                if shape.shape[0] >= 46:  # 68-point
                    sub = shape[idx]
                elif shape.shape[0] == 5:
                    # Map 5-point to eye corners + nose proxy: [left_eye_outer, left_eye_inner, right_eye_inner, right_eye_outer, nose_tip]
                    # Build pseudo 6 points by duplicating nose for corners (approx)
                    l_outer, l_inner, r_inner, r_outer, nose = shape
                    sub = np.array([l_inner, l_outer, r_inner, r_outer, nose, nose], dtype=np.float32)
                else:
                    tracked_pts = None
                    continue
                tracked_pts = sub.astype(np.float32)
        else:
            # Track previous landmarks with optical flow
            if prev_gray is not None and tracked_pts is not None:
                next_pts, old_pts, st = track_landmarks_optical_flow(prev_gray, gray, tracked_pts.reshape(-1,1,2))
                if next_pts.shape[0] >= 6:
                    tracked_pts = next_pts.astype(np.float32)
                else:
                    tracked_pts = None  # force re-detect

        # Draw landmarks and proceed if we have them
        if tracked_pts is not None and tracked_pts.shape[0] >= 6:
            # Smooth landmarks
            smoothed = lmk_ema.update(tracked_pts)
            for (x, y) in smoothed.astype(int):
                cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)

            # Prepare for head pose & normalization
            landmarks_sub = smoothed.reshape(6, 1, 2).astype(np.float32)
            hr, ht = estimate_head_pose(landmarks_sub, facePts, camera_matrix, camera_distortion)
            face_patch, lmks_norm, hr_norm = normalize_face(orig, face_model, landmarks_sub, hr, ht, camera_matrix)

            # Gaze inference
            with torch.no_grad():
                inp = face_patch[:, :, [2, 1, 0]]
                inp = trans(inp)
                if use_cuda:
                    inp = inp.float().cuda().unsqueeze(0)
                else:
                    inp = inp.float().unsqueeze(0)
                pred = model(inp)[0]
                if use_cuda:
                    pred_np = pred.detach().cpu().numpy()
                else:
                    pred_np = pred.detach().numpy()

            # Smooth gaze
            pred_np = gaze_ema.update(pred_np)

            # Visualize on face patch and on frame (optional)
            vis_patch = face_patch.copy()
            for (x,y) in lmks_norm.astype(int):
                cv2.circle(vis_patch, (int(x), int(y)), 2, (0,255,0), -1)
            vis_patch = draw_gaze(vis_patch, pred_np)

            # Show mini window of normalized patch
            small_patch = cv2.resize(vis_patch, (224,224))
            frame[0:224, 0:224] = small_patch

        # FPS
        t1 = time.time()
        fps = 1.0 / max(1e-6, (t1 - t0))
        fps_hist.append(fps)
        avg_fps = sum(fps_hist) / len(fps_hist)
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow('ETH-XGaze Realtime', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        prev_gray = gray.copy()
        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
