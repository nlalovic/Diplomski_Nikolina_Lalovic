#SVE PODESIVE VRIJEDNOSTI SISTEMA NA JEDNOM MJESTU

import os as _os

import numpy as np

_OVDJE = _os.path.dirname(_os.path.abspath(__file__))

# PUPIL CAPTURE

PUPIL_IP = "127.0.0.1"  
PUPIL_REQ_PORT = 50020 #fabricki port Pupil Remote za Network API
GAZE_CONFIDENCE_MIN = 0.6 #minimalna pouzdanost tacke pogleda
#izgled oznake pogleda na slici 
GAZE_CIRCLE_RADIUS = 20
GAZE_COLOR = (0, 255, 0)

# YOLO

YOLO_MODEL = _os.path.join(_OVDJE, "best.pt") #sopstveni model (plisana igracka, dzojstik, solja)
YOLO_IMG_SIZE = 640 #rezolucija slike za YOLO
YOLO_EVERY_NTH_FRAME = 3 #na koliko kadrova se pokrece YOLO
YOLO_CONF_MIN = 0.4 #minimalna pouzdanost detekcije
#izgled okvira oko detektovanih predmeta
BBOX_COLOR = (0, 255, 0)            #zeleno obican, detektovan predmet
BBOX_THICKNESS = 2
BBOX_HIGHLIGHT_COLOR = (0, 0, 255)  #crveno predmet u koji trenutno gledamo
BBOX_HIGHLIGHT_THICKNESS = 4


# DWELL

DWELL_THRESHOLD = 2.0 #prag zadrzavanja pogleda u sekundama
DWELL_GRACE_PERIOD = 0.3 #tolerancija za prekid pogleda u sekundama
DWELL_LOOK_ANNOUNCE_S = 1.0 #nakon koliko sekundi se vrsi ispis na konzolu
#progress bar dwell
DWELL_BAR_W = 250
DWELL_BAR_H = 25
DWELL_BAR_X = 10
DWELL_BAR_Y = 10

# ISPIS U KONZOLU

# "demo"  kratak ispis, za snimanje demonstracija
# "debug"  sve - koordinate, korekcije, visine
LOG_MODE = "demo"

#ARUCO

ARUCO_DICT = "DICT_4X4_100" 

OBJECT_MARKER_LENGTH = 0.05  
TABLE_MARKER_LENGTH = 0.069   #duzina crne ivice referentnih markera u metrima
OBJECT_MARKER_LENGTHS = {
    1: 0.040,   #solja
    2: 0.040,   #dzojstik
    3: 0.040,   #plisana igracka
}

TABLE_MARKER_IDS = [24, 23, 22, 21] # BL BR TR TL
TABLE_MARKER_CENTERS = {
    24: (0.000, 0.000),   #donji-lijevi koordinatn pocetak k.s. radne povrsine
    23: (0.804, 0.000),   #donji-desni X+
    22: (0.804, 0.604),   #gornji-desni
    21: (0.000, 0.604),   #gornji-lijevi Y+
}
MARKER_MEMORY_TTL = 0.5 #kratka memorija markera u sekundama

POSE_ERROR_RATIO = 10.0
POSE_NORMAL_MIN_DOT = 0.5



# PREDMETI

#ime   naziv klase kao u YOLO modelu 
#w_mm  sirina za hvat
#h_mm       visina predmeta od stola do vrha
#grip_z_mm  visina hvata iznad stola 
#close_mm   koliko se zatvaraju prsti pri podizanju i prenosenjy
#grip_dx_mm/grip_dy_mm offset ako se predmet ne hvata na mjestu markera

OBJECT_SIZES = {
    1: {"ime": "cup",      "w_mm": 70.0,  "h_mm": 185.0, "grip_z_mm": 170.0,
        "close_mm": 67.0, "grip_dx_mm": 0.0, "grip_dy_mm": -10.0}, 
    2: {"ime": "joystick", "w_mm": 70.0,  "h_mm": 40.0,  "grip_z_mm": 32.0,
        "close_mm": 65.0, "grip_dx_mm": 0.0, "grip_dy_mm": -15.0},  
    3: {"ime": "cow",      "w_mm": 50.0,  "h_mm": 50.0,  "grip_z_mm": 25.0,
        "close_mm": 53.0, "grip_dx_mm": 0.0, "grip_dy_mm": -10.0}, 
}

OBJECT_MARKER_IDS = list(OBJECT_SIZES.keys()) #ID-jevi predmeta
OBJECT_NAMES = {mid: v["ime"] for mid, v in OBJECT_SIZES.items()} #klasa predmeta

# REZIMI RADA RUKE
ARM_MODE = "HVAT"
HOVER_HEIGHT_M = 0.05 #koliko iznad tacke hvata ruka stane u HOVER rezimu u metrima

# VISINE

TABLE_Z_ROBOT_MM = -150.0 #Z POVRSINE STOLA u frejmu BAZE ROBOTA u mm

#grip_z fiksna izmjerena visina grip_z_mm iz OBJECT_SIZES
#marker  Z markera koji je ArUco izmjerio u tom trenutku minus GRASP_INSERT_MM
GRASP_REFERENCE = "marker"

#Koliko prsti sidju ispod z ose markera pri hvatu u mm
GRASP_INSERT_MM = {
    1: 30.0,   # cup
    2: 30.0,   # joystick
    3: 35.0,   # cow
}
GRASP_INSERT_DEFAULT_MM = 30.0   #za predmet kojeg nema u tabeli gore

#SIGURNOSNI POD ZA HVAT vrh prstiju nikad nize od ovoliko mm IZNAD POVRSINE STOLA
GRASP_MIN_ABOVE_TABLE_MM = 5.0

#Koliko se predmet digne od stola poslije stiskanja, prije nosenja (mm)
LIFT_AFTER_GRASP_MM = 70.0

#koliko NAJMANJE iznad stola smije biti ruka dok nosi predmet 
CARRY_MIN_ABOVE_TABLE_MM = 70.0

#koliko vise od visine hvata se predmet pusti pri odlaganju (mm)
RELEASE_ABOVE_PICK_MM = 5.0

#koliko se ruka digne poslije pustanja, prije nego sto se skloni (mm).
RETREAT_AFTER_RELEASE_MM = 100.0

# ODLAGANJE 

DROP_TABLE_POINTS = [
    (0.130, 0.100),   #zona 1
    (0.130, 0.325),   #zona 2
    (0.250, 0.325),   #zona 3
]
DROP_YAW_TABLE_DEG = 0.0

CORR_TABLE_X_MM = 0.0 #fiksne korekcije po x i y
CORR_TABLE_Y_MM = 0.0

IDLE_RETURN_S = 25.0 #nakon koliko vremena bez selekcije se ruka vraca u inicijalni polozaj
ASINHRONA_RUKA = True #da bi slika bil akontinualna mora biti True ako nije pri pomjeranju ruke se zamrzava sika

# SNIMANJE DEMO

SNIMANJE = True
SNIMAK_FOLDER = _os.path.join(_OVDJE, "snimci")
SNIMAK_FPS = 30.0
SNIMAK_MAX_RUPA_S = 2.0 #najduza rupa koja se popunjava zadnjim kadrom
SNIMAK_OVERLAY_LINIJA = 0 #koliko poruka se ispisuje u prozoru sa slikom 
SNIMAK_TITLOVI = True #ako je True pravi se vremenski uskladjen .srt sa porukama

# KALIBRACIJA STO -> ROBOT

T_ROBOT_TABLE = np.array([                            # rezultati klaibracije
    [-0.004168, -0.999991,  0.000000,  0.546360],
    [ 0.999991, -0.004168,  0.000000, -0.658666],
    [ 0.000000,  0.000000,  1.000000, -0.149620],
    [ 0.000000,  0.000000,  0.000000,  1.000000]
], dtype=np.float64)

# KALIBRACIJA SCENE KAMERE 

#oblik matrice [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
#fx, fy  zizna daljina u pikselima
#cx, cy  principal point (gdje opticka osa probada sliku)
CAMERA_MATRIX = np.array([
    [798.7471905384108, 0.0,                608.760172686715],
    [0.0,               798.4705057253415,  344.2021065407961],
    [0.0,               0.0,                1.0],
], dtype=np.float64)

#koeficijenti distorzije [k1, k2, p1, p2, k3, k4, k5, k6].
#k radijalna distorzija
#p tangencijalna distorzija
DIST_COEFFS = np.array([
    0.971987987417899,
    0.4518184659170611,
    0.0002491351311354695,
    -0.00012145504283113181,
    0.03832729825368759,
    1.4404720418430035,
    0.8037125580700881,
    0.18770284993026337,
], dtype=np.float64)
