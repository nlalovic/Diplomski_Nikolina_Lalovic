import math
import cv2
import numpy as np

def make_transform(rvec, tvec): # vektor rotacije rvec, vektor translacije tvec iz solvePnP
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64)) # pretvara vektor rotacije u matricu rotacije 3x3
    t = np.asarray(tvec, dtype=np.float64).reshape(3) # pretvara tvec u jednodimenzionalni niz
    T = np.eye(4) # pravi jedinicnu matricu
    T[:3, :3] = R # upisuje matricu rotacije 3x3 u prve tri vrste i prve tri kolone
    T[:3, 3] = t # upisuje vektor translacije u prve tri vrste cetvrte kolone
    return T


def invert_transform(T): # inverzija matrice
    R = T[:3, :3] 
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T # upisuje trasnponovanu matricu rotacije 
    Ti[:3, 3] = -R.T @ t # vektor translacije inevrzne matrice se upisuje kao proizvod -transponovane matrice rotacije i vektora translacije
    return Ti


def compose(*transforms): # ulancavanje matrica
    out = np.eye(4)
    for T in transforms:
        out = out @ T
    return out


def transform_point(T, p): # transformcija tacke iz jednog k.s. u drugi
    p = np.asarray(p, dtype=np.float64).reshape(3)
    ph = np.append([p, 1.0])  # tacka (x,y,z) se pretvara u (x,y,z,1) kako bi se moglo mnoziti sa matricom 4x4
    return (T @ ph)[:3]  # dodata jeidnica se izbacuje iz rezultata


# IZ MATRICE U BROJEVE KOJE RUKA RAZUMIJE

def rotation_to_euler_zyx(R):
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0
    return roll, pitch, yaw


def transform_to_xyzrpy(T, degrees=True):
    x, y, z = T[:3, 3]
    roll, pitch, yaw = rotation_to_euler_zyx(T[:3, :3])
    if degrees:
        roll, pitch, yaw = (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))
    return float(x), float(y), float(z), float(roll), float(pitch), float(yaw)


# USREDNJAVANJE POZA 

def geodesic_angle(Ra, Rb):
    Rr = Ra.T @ Rb
    c = (np.trace(Rr) - 1.0) / 2.0
    c = max(-1.0, min(1.0, c))    
    return math.acos(c)


def average_rotations(rotations):
    # usrednjavanje rotacija
    M = np.zeros((3, 3))
    for R in rotations:
        M += R
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    return R


def average_poses(transforms, max_trans_dev=0.05, max_rot_deg=25.0):
    n = len(transforms)
    if n == 0:
        return None, 0, 0

    Ts = np.array([T[:3, 3] for T in transforms])    # (n, 3) translacije
    Rs = [T[:3, :3] for T in transforms]             # lista rotacija

    # 1. translacioni - blizu medijane
    med = np.median(Ts, axis=0)
    trans_ok = np.linalg.norm(Ts - med, axis=1) <= max_trans_dev

    # 2. rotacioni - blizu medoida
    ang = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            a = geodesic_angle(Rs[i], Rs[j])
            ang[i, j] = ang[j, i] = a
    medoid = int(np.argmin(np.median(ang, axis=1)))
    rot_ok = np.array([geodesic_angle(Rs[medoid], Rs[k]) <= math.radians(max_rot_deg)
                       for k in range(n)])

    inl = trans_ok & rot_ok
    if not inl.any():
        inl = np.ones(n, dtype=bool)

    # 3. usrednji preostale
    t_avg = Ts[inl].mean(axis=0)
    R_avg = average_rotations([Rs[k] for k in range(n) if inl[k]])

    T = np.eye(4)
    T[:3, :3] = R_avg
    T[:3, 3] = t_avg
    return T, int(inl.sum()), n
