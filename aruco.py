import math

import cv2
import numpy as np

import transformacije as tf
from config import (
    ARUCO_DICT,
    CAMERA_MATRIX,
    DIST_COEFFS,
    MARKER_MEMORY_TTL,
    OBJECT_MARKER_LENGTH,
    TABLE_MARKER_LENGTH,
    TABLE_MARKER_IDS,
    TABLE_MARKER_CENTERS,
    OBJECT_MARKER_LENGTHS,
    POSE_ERROR_RATIO,
    POSE_NORMAL_MIN_DOT,
)

_DICT_MAP = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100
}


def length_for_id(marker_id):
    if marker_id in OBJECT_MARKER_LENGTHS:
        return OBJECT_MARKER_LENGTHS[marker_id]
    if marker_id in TABLE_MARKER_IDS:
        return TABLE_MARKER_LENGTH
    return OBJECT_MARKER_LENGTH   # nepoznat ID rezervna velicina iz config-a


def setup_detector(): # pravljenje detektora, poziva se jednom na startu programa
    aruco_dict = cv2.aruco.getPredefinedDictionary(_DICT_MAP[ARUCO_DICT])
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX # dotjerivanje na subpixels, veca tacnost
    return cv2.aruco.ArucoDetector(aruco_dict, params)


def detect_markers(frame, detector): # vraca (corners,ids) svih detektovanih markera
    corners, ids, _odbaceni = detector.detectMarkers(frame)
    return corners, ids


def marker_center(marker_corners): # centar markera u slici u pikselima, za fuziju sa YOLO
    pts = marker_corners.reshape(4, 2)       # (1,4,2) -> (4,2)
    return int(pts[:, 0].mean()), int(pts[:, 1].mean())


def distance_meters(tvec): # duzina vektora translacije u metrima
    x, y, z = tvec.flatten()  # (3,1) -> [x, y, z]
    return math.sqrt(x ** 2 + y ** 2 + z ** 2)

# POZA MARKERA, uz razrjesavanje flipovanja
FLIP_STATS = {"ukupno": 0, "nerazlucivo": 0, "preokrenuto": 0}


def table_normal(table_pose): # z osa u k.s. kamere
    if table_pose is None:
        return None
    R, _ = cv2.Rodrigues(np.asarray(table_pose["rvec"], dtype=np.float64)) # pretvara vektor rotacije u matricu rotacije
    return R[:, 2]  # treca kolona rotacione matrice - Z osa


def estimate_pose_disambiguated(marker_corners, marker_length, ref_normal=None):
    #Poza jednog markera (rvec, tvec) u frejmu KAMERE, sa svjesnim izborom
    
    # Gdje se uglovi markera nalaze u njegovom SOPSTVENOM koordinatnom sistemu
    half = marker_length / 2.0
    object_points = np.array([
        [-half,  half, 0],  # TL
        [ half,  half, 0],  # TR
        [ half, -half, 0],  # BR
        [-half, -half, 0],  # BL
    ], dtype=np.float32)
    image_points = marker_corners.reshape(4, 2).astype(np.float32)

    n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
        object_points, image_points,
        CAMERA_MATRIX, DIST_COEFFS,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if n == 0:
        return None, None

    if errs is not None and len(errs) == n:
        e = [float(np.ravel(errs[k])[0]) for k in range(n)]
    else:
        e = [0.0] * n
    order = sorted(range(n), key=lambda k: e[k])
    best = order[0] # rjesenje sa najmanjom greskom

    if n == 1 or ref_normal is None:
        return rvecs[best], tvecs[best]

    FLIP_STATS["ukupno"] += 1

    second = order[1]
    if e[best] * POSE_ERROR_RATIO < e[second]:
        return rvecs[best], tvecs[best]

    FLIP_STATS["nerazlucivo"] += 1
    ref = np.asarray(ref_normal, dtype=np.float64).reshape(3)
    dots = []
    for k in range(n):
        R, _ = cv2.Rodrigues(rvecs[k])
        dots.append(float(np.dot(R[:, 2], ref)))
    pick = int(np.argmax(dots))

    if dots[pick] < POSE_NORMAL_MIN_DOT:
        return rvecs[best], tvecs[best]

    if pick != best:
        FLIP_STATS["preokrenuto"] += 1  
    return rvecs[pick], tvecs[pick]


def detect_and_estimate(frame, detector, ref_normal=None):
    # nadji sve markere na kadru i svakom izracunaj pozu.
    corners, ids = detect_markers(frame, detector)

    markers = []
    if ids is None:
        return markers

    for i in range(len(ids)):
        c = corners[i]
        mid = int(ids[i][0])
        rvec, tvec = estimate_pose_disambiguated(c, length_for_id(mid), ref_normal)
        markers.append({
            "id": mid,
            "center": marker_center(c), # piksel (x, y)
            "rvec": rvec, # rotacija u frejmu kamere
            "tvec": tvec, # translacija u frejmu kamere
            "distance": distance_meters(tvec) if tvec is not None else None,
            "corners": c,
        })
    return markers


class MarkerMemory:

    def __init__(self, ttl=MARKER_MEMORY_TTL):
        self.ttl = ttl
        self._seen = {}   # id markera 

    def update(self, markers, now):
        for m in markers:
            self._seen[m["id"]] = (m, now)

        # zaboravi one koje nismo vidjeli duze od ttl sekundi
        self._seen = {
            mid: (m, t)
            for mid, (m, t) in self._seen.items()
            if now - t <= self.ttl
        }

    def get_markers(self):
        return [m for (m, _t) in self._seen.values()]


def find_marker_by_id(markers, marker_id):
    for m in markers:
        if m["id"] == marker_id:
            return m
    return None

# SISTEM STOLA

def make_table_board():
    aruco_dict = cv2.aruco.getPredefinedDictionary(_DICT_MAP[ARUCO_DICT])
    half = TABLE_MARKER_LENGTH / 2.0

    obj_points = []
    ids = []
    for mid in TABLE_MARKER_IDS:
        cx, cy = TABLE_MARKER_CENTERS[mid]   # centar markera na stolu (metri)
        corners = np.array([
            [cx - half, cy + half, 0.0],  # TL
            [cx + half, cy + half, 0.0],  # TR
            [cx + half, cy - half, 0.0],  # BR
            [cx - half, cy - half, 0.0],  # BL
        ], dtype=np.float32)
        obj_points.append(corners)
        ids.append(mid)

    return cv2.aruco.Board(obj_points, aruco_dict, np.array(ids, dtype=np.int32))

_MIN_TABLE_POINTS = 12


def estimate_table_pose(markers, board):
    table = [m for m in markers if m["id"] in TABLE_MARKER_IDS]
    if not table:
        return None

    corners = [m["corners"] for m in table]              # 2D uglovi na slici
    ids = np.array([[m["id"]] for m in table], dtype=np.int32)

    # board upari 2D uglove sa njihovim poznatim 3D pozicijama na stolu
    obj_pts, img_pts = board.matchImagePoints(corners, ids)
    if obj_pts is None or len(obj_pts) < _MIN_TABLE_POINTS:
        return None

    # jedan solvePnP iz svih tih tacaka odjednom -> poza cijelog stola
    ok, rvec, tvec = cv2.solvePnP(
        obj_pts, img_pts, CAMERA_MATRIX, DIST_COEFFS,
        flags=cv2.SOLVEPNP_IPPE,
    )
    if not ok:
        return None

    return {"rvec": rvec, "tvec": tvec}


def object_pose_in_table_frame(obj_marker, table_pose):
    T_cam_table = tf.make_transform(table_pose["rvec"], table_pose["tvec"])
    T_cam_obj = tf.make_transform(obj_marker["rvec"], obj_marker["tvec"])
    return tf.compose(tf.invert_transform(T_cam_table), T_cam_obj)


class TablePoseAccumulator:
    #usrednjavanje poze

    def __init__(self, window):
        self.window = window # koliko sekundi unazad pamtimo (dwell)
        self._buf = {} # id markera -> lista [(vrijeme, T_sto_predmet)]

    def update(self, markers, table_pose, now):
        # bez vidljivog stola ne mozemo u frejm stola -> taj kadar preskacemo
        if table_pose is None:
            return
        for m in markers:
            # zanimaju nas samo markeri PREDMETA (ne stola) sa ispravnom pozom
            if m["id"] in TABLE_MARKER_IDS or m["rvec"] is None or m["tvec"] is None:
                continue
            self._buf.setdefault(m["id"], []).append(
                (now, object_pose_in_table_frame(m, table_pose)))

        # izbaci uzorke starije od prozora; ocisti prazne unose
        for mid in list(self._buf.keys()):
            self._buf[mid] = [(t, T) for (t, T) in self._buf[mid]
                              if now - t <= self.window]
            if not self._buf[mid]:
                del self._buf[mid]

    def averaged(self, marker_id, now):
        items = self._buf.get(marker_id, [])
        Ts = [T for (t, T) in items if now - t <= self.window]
        if not Ts:
            return None, 0, 0
        return tf.average_poses(Ts)


def draw_detected(frame, markers):
    for m in markers:
        cv2.aruco.drawDetectedMarkers(frame, [m["corners"]], np.array([[m["id"]]]))
        if m["rvec"] is not None:
            cv2.drawFrameAxes(frame, CAMERA_MATRIX, DIST_COEFFS,
                              m["rvec"], m["tvec"], length_for_id(m["id"]) * 0.5)
