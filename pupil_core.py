#GLAVNI PROGRAM 
import math
import os
import re
import threading
import time
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

import pupilcore
import detection
import dwell
import aruco
import transformacije as tf
import xarm_kontrola as xk

from config import (
    YOLO_MODEL, YOLO_IMG_SIZE, YOLO_EVERY_NTH_FRAME, YOLO_CONF_MIN,
    GAZE_CIRCLE_RADIUS, GAZE_COLOR,
    BBOX_HIGHLIGHT_COLOR, BBOX_HIGHLIGHT_THICKNESS,
    DWELL_THRESHOLD, DWELL_LOOK_ANNOUNCE_S,
    T_ROBOT_TABLE, TABLE_Z_ROBOT_MM,
    OBJECT_SIZES,
    ARM_MODE, HOVER_HEIGHT_M,
    CORR_TABLE_X_MM, CORR_TABLE_Y_MM,
    DROP_TABLE_POINTS, DROP_YAW_TABLE_DEG,
    GRASP_REFERENCE, GRASP_INSERT_MM, GRASP_INSERT_DEFAULT_MM,
    GRASP_MIN_ABOVE_TABLE_MM, IDLE_RETURN_S,
    LOG_MODE,
    SNIMANJE, SNIMAK_FOLDER, SNIMAK_FPS, SNIMAK_MAX_RUPA_S,
    SNIMAK_OVERLAY_LINIJA, SNIMAK_TITLOVI, ASINHRONA_RUKA,
)


DEBUG_LOG = (LOG_MODE == "debug")
LOG_LINIJE = deque(maxlen=max(1, SNIMAK_OVERLAY_LINIJA))

# (vrijeme, tekst) svake poruke - .srt 
LOG_SA_VREMENOM = []
CRTA = "-" * 62


def demo(*args, **kwargs):
    print(*args, **kwargs)
    if not args:
        return
    tekst = " ".join(str(a) for a in args).strip()
    if not tekst or set(tekst) == {"-"}:
        return
    if SNIMAK_OVERLAY_LINIJA > 0:
        LOG_LINIJE.append(tekst)
    if SNIMANJE and SNIMAK_TITLOVI:
        LOG_SA_VREMENOM.append((time.time(), tekst))


def dbg(*args, **kwargs):
    if DEBUG_LOG:
        print(*args, **kwargs)


def crta():
    demo(CRTA)


# SNIMANJE DEMOA
class Snimac:
    TITL_PRESKOCI = ("[snimak]",)

    def __init__(self):
        self.writer = None
        self.putanja = None
        self.osnova = None        
        self.t0 = None            # trenutak otvaranja fajla
        self.n_upisanih = 0
        self.odustali = False

    def _otvori(self, frame):
        os.makedirs(SNIMAK_FOLDER, exist_ok=True)
        ime = time.strftime("demo_%Y-%m-%d_%H-%M-%S")
        h, w = frame.shape[:2]
        for fourcc, ext in (("mp4v", ".mp4"), ("XVID", ".avi")):
            putanja = os.path.join(SNIMAK_FOLDER, ime + ext)
            wr = cv2.VideoWriter(putanja, cv2.VideoWriter_fourcc(*fourcc),
                                 SNIMAK_FPS, (w, h))
            if wr.isOpened():
                self.writer = wr
                self.putanja = putanja
                self.osnova = os.path.join(SNIMAK_FOLDER, ime)
                self.t0 = time.time()
                self.n_upisanih = 0
                demo(f"[snimak]   Snimam u {putanja}  ({w}x{h}, {SNIMAK_FPS:.0f} fps)")
                return True
        demo("[!]        Ne moze se otvoriti video fajl.")
        return False

    def dodaj(self, frame):
        if self.odustali:
            return
        if self.writer is None:
            if not self._otvori(frame):
                self.odustali = True
                return

        # koliko je kadrova trebalo da bude upisano do OVOG trenutka
        treba = int(round((time.time() - self.t0) * SNIMAK_FPS))
        n = treba - self.n_upisanih
        if n <= 0:
            return                                            # petlja brza od fps-a
        n = min(n, int(SNIMAK_MAX_RUPA_S * SNIMAK_FPS))       # ne popunjavaj ogromne rupe
        for _ in range(n):
            self.writer.write(frame)
        self.n_upisanih += n

    @staticmethod
    def _za_titl(tekst):
        if tekst.startswith(Snimac.TITL_PRESKOCI):
            return None
        return re.sub(r"\s{2,}", " ", tekst).strip() or None

    def _upisi_titl(self):
        poruke = []
        for t, tekst in LOG_SA_VREMENOM:
            if t < self.t0:
                continue                      # poruka od prije pocetka snimanja
            cist = self._za_titl(tekst)
            if cist:
                poruke.append((t - self.t0, cist))
        if not poruke:
            return
        kraj_snimka = time.time() - self.t0

        def vrijeme(sek):
            #sekunde -> format koji .srt ocekuje: HH:MM:SS,mmm
            sek = max(0.0, sek)
            h = int(sek // 3600)
            m = int((sek % 3600) // 60)
            s_ = int(sek % 60)
            ms = int(round((sek - int(sek)) * 1000))
            return f"{h:02d}:{m:02d}:{s_:02d},{ms:03d}"
        GRUPA_S = 1.5  
        MAX_REDOVA = 3      
        grupe = []
        for t, tekst in poruke:
            if (grupe and t - grupe[-1][0] <= GRUPA_S
                    and len(grupe[-1][1]) < MAX_REDOVA):
                grupe[-1][1].append(tekst)
            else:
                grupe.append((t, [tekst]))

        MAX_TRAJANJE = 6.0
        redovi = []
        for i, (t, linije) in enumerate(grupe):
            sljedeci = grupe[i + 1][0] if i + 1 < len(grupe) else kraj_snimka + 2.0
            do = min(sljedeci, t + MAX_TRAJANJE)
            if do <= t:
                do = t + 0.5            
            redovi.append(
                f"{i + 1}\n{vrijeme(t)} --> {vrijeme(do)}\n"
                + "\n".join(linije) + "\n")

        putanja = self.osnova + ".srt"
        with open(putanja, "w", encoding="utf-8") as f:
            f.write("\n".join(redovi))
        demo(f"[snimak]   Titl: {putanja}  ({len(poruke)} poruka)")

    def zatvori(self):
        if self.writer is None:
            return
        trajanje = self.n_upisanih / SNIMAK_FPS
        self.writer.release()
        self.writer = None
        demo(f"[snimak]   Zavrseno: {self.putanja}  "
             f"({self.n_upisanih} kadrova, {trajanje:.1f} s)")
        if SNIMAK_TITLOVI:
            self._upisi_titl()


def nacrtaj_log(frame):
    if SNIMAK_OVERLAY_LINIJA <= 0 or not LOG_LINIJE:
        return
    h, w = frame.shape[:2]
    linije = list(LOG_LINIJE)
    vis = 22 * len(linije) + 14
    y0 = h - vis - 34                      # iznad reda sa stanjem odlagalista

    podloga = frame[y0:y0 + vis, 0:w].copy()
    frame[y0:y0 + vis, 0:w] = cv2.addWeighted(
        podloga, 0.35, np.zeros_like(podloga), 0.65, 0)

    for i, linija in enumerate(linije):
        #zadnja poruka je najsvjezija
        boja = (255, 255, 255) if i == len(linije) - 1 else (170, 170, 170)
        cv2.putText(frame, linija[:96], (10, y0 + 20 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, boja, 1, cv2.LINE_AA)


# RACUN VISINA I UGLOVA prevod iz sistema stola u sistem robota
def table_z_mm():
    if TABLE_Z_ROBOT_MM is not None:
        return float(TABLE_Z_ROBOT_MM)
    return float(tf.transform_point(T_ROBOT_TABLE, (0.0, 0.0, 0.0))[2]) * 1000.0


def drop_yaw_robot_deg():
    if DROP_YAW_TABLE_DEG is None:
        return None
    th = math.radians(DROP_YAW_TABLE_DEG)
    Rz = np.array([[math.cos(th), -math.sin(th), 0.0],
                   [math.sin(th),  math.cos(th), 0.0],
                   [0.0,           0.0,          1.0]])
    T = np.eye(4)
    T[:3, :3] = T_ROBOT_TABLE[:3, :3] @ Rz
    return tf.transform_to_xyzrpy(T)[5]


def grasp_z_mm(marker_id, marker_z_mm):
    if GRASP_REFERENCE == "grip_z":
        gz = (OBJECT_SIZES.get(marker_id) or {}).get("grip_z_mm")
        if gz is not None:
            return table_z_mm() + gz, f"fiksno {gz:.0f} mm iznad stola (izmjereno)"
        # predmet nema izmjerenu konstantu -> nema izbora, padamo na viziju
        ins = GRASP_INSERT_MM.get(marker_id, GRASP_INSERT_DEFAULT_MM)
        return (marker_z_mm - ins,
                f"Z markera - {ins:.0f} mm (grip_z_mm nije zadan!)")
    ins = GRASP_INSERT_MM.get(marker_id, GRASP_INSERT_DEFAULT_MM)
    return marker_z_mm - ins, f"Z markera iz vizije - {ins:.0f} mm"

# ODLAGALISTA
class DropZones:
    def __init__(self, points):
        self.points = points   # lista (X, Y) u sistemu stola
        self.index = 0         # koliko ih je do sada popunjeno

    def sve_popunjeno(self):
        return self.index >= len(self.points)

    def sljedeca(self):
        # Vrati (broj_zone, (X, Y)) za prvo slobodno odlagaliste
        if self.sve_popunjeno():
            return None, None
        return self.index + 1, self.points[self.index]

    def zauzmi(self):
        # Oznaci trenutno odlagaliste kao popunjeno i predji na sljedece
        self.index += 1

    def reset(self):
        # Sve zone opet slobodne 
        self.index = 0

    def status(self):
        return f"Odlagalista: {self.index}/{len(self.points)} popunjeno"

# SLANJE KOMANDE RUCI

def setup_arm():
    # povezi se na xArm - samo ako ARM_MODE nije MOCK
    if ARM_MODE == "MOCK":
        dbg("[xArm] ARM_MODE=MOCK -> ruka se NE dira, samo ispis komandi.")
        return None

    dbg(f"[xArm] ARM_MODE={ARM_MODE} -> povezujem se na ruku...", flush=True)
    arm = xk.XArmController(quiet=not DEBUG_LOG)  
    arm.connect()
    return arm


def send_to_xarm(arm, zones, obj_name, marker_id, T_table_obj, n_inliers, n_total):
    # Prima usrednjenu pozu predmeta u sistemu STOLA i salje ruci gdje da ide
    # --- slucajevi u kojima nemamo dovoljno podataka da bilo sta posaljemo ---
    if marker_id is None:
        demo(f"[!]        '{obj_name}' selektovan, ali nema markera na njemu "
             f"-> ne znam gdje je, preskacem.")
        return
    if T_table_obj is None:
        demo(f"[!]        '{obj_name}': sto se nije vidio tokom gledanja "
             f"-> ne mogu izracunati poziciju, preskacem.")
        return

    # KOREKCIJA CILJA 
    
    dims_obj = OBJECT_SIZES.get(marker_id) or {}
    dx_mm = float(dims_obj.get("grip_dx_mm", 0.0))
    dy_mm = float(dims_obj.get("grip_dy_mm", 0.0))
    sx, sy, sz, sr, sp, syaw = tf.transform_to_xyzrpy(T_table_obj)

    yaw_rad = math.radians(syaw)
    cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
    off_x_mm = cos_y * dx_mm - sin_y * dy_mm
    off_y_mm = sin_y * dx_mm + cos_y * dy_mm

    tot_x_mm = off_x_mm + CORR_TABLE_X_MM
    tot_y_mm = off_y_mm + CORR_TABLE_Y_MM

    T_table_corr = T_table_obj.copy()
    T_table_corr[0, 3] += tot_x_mm / 1000.0   # mm -> m
    T_table_corr[1, 3] += tot_y_mm / 1000.0

    # u k.s. ROBOTA: T_robot_predmet = T_robot_sto @ T_sto_predmet
    T_robot_obj = tf.compose(T_ROBOT_TABLE, T_table_corr)

    # pozicija u metrima, uglovi u stepenima
    rx, ry, rz, rr, rp, ryaw = tf.transform_to_xyzrpy(T_robot_obj)
    x_mm, y_mm, z_mm = rx * 1000.0, ry * 1000.0, rz * 1000.0

    # visina hvata 
    gz, gz_opis = grasp_z_mm(marker_id, z_mm)

    # demo poruka
    demo(f"[izbor]    {obj_name} selektovan.")
    demo(f"[cilj]     X={T_table_corr[0, 3]*100:+.1f} cm  Y={T_table_corr[1, 3]*100:+.1f} cm "
         f"(sistem stola)")

    # koje je odlagaliste na redu
    zone_no, zone_xy = zones.sljedeca()
    if zone_no is None:
        demo(f"[!]        Sva {len(zones.points)} odlagalista su popunjena -> nemam gdje "
             f"da odlozim '{obj_name}', preskacem.")
        demo("           (pritisni 'r' u prozoru da ih oslobodis za novi snimak)")
        return

    # MOCK
    if arm is None:
        demo(f"[MOCK]     Predmet spusten u zonu za odlaganje {zone_no}.")
        zones.zauzmi()
        dbg(f"    {zones.status()}")
        demo("[sistem]   Cekam novu komandu.")
        return

    # HOVER
    if ARM_MODE == "HOVER":
        hz_mm = gz + HOVER_HEIGHT_M * 1000.0
        dbg(f"    -> HOVER {HOVER_HEIGHT_M*100:.0f} cm iznad tacke hvata "
            f"({gz_opis}; hvat z={gz:.0f}, hover z={hz_mm:.0f} mm)")
        try:
            arm.move_topdown(x_mm, y_mm, hz_mm, ryaw, speed=xk.SPEED)
            demo(f"[hover]    Ruka je iznad predmeta, Z="
                 f"{(hz_mm - table_z_mm())/10:.1f} cm iznad stola.")
        except Exception as e:
            # npr. tacka van dohvata ne rusi petlju, samo javi i nastavi
            demo(f"[!]        pokret nije uspio: {e}")
        demo("[sistem]   Cekam novu komandu.")
        return

    #HVAT
    dims = OBJECT_SIZES.get(marker_id)

    drop_robot = tf.transform_point(
        T_ROBOT_TABLE, (zone_xy[0], zone_xy[1], 0.0)
    ) * 1000.0
    drop_xyz = (float(drop_robot[0]), float(drop_robot[1]), table_z_mm())

    dbg(f"    -> HVAT: sirina {dims['w_mm']:.0f} visina {dims['h_mm']:.0f} mm, "
        f"yaw={ryaw:+.1f}, ZONA {zone_no} @ sto "
        f"(X={zone_xy[0]:+.3f}, Y={zone_xy[1]:+.3f}) m -> robot "
        f"({drop_xyz[0]:.0f},{drop_xyz[1]:.0f},{drop_xyz[2]:.0f}) mm")
    dbg(f"       visina hvata: {gz_opis} -> z={gz:.0f} mm; "
        f"stisak {dims.get('close_mm', 0):.0f} mm")

    def na_dogadjaj(sta, _z_mm):
        if sta == "uhvacen":
            demo("[hvat]     Predmet uhvacen!")
        elif sta == "prenosim":
            demo(f"[nosim]    Prenosim predmet na odlagaliste {zone_no}.")
        elif sta == "spusten":
            demo(f"[odlazem]  Predmet spusten u zonu {zone_no}.")

    try:
        arm.pick_top_down(x_mm, y_mm, gz, ryaw,
                          width_mm=dims["w_mm"], height_mm=dims["h_mm"],
                          drop_xyz=drop_xyz,
                          close_mm=dims.get("close_mm"),
                          hover_mm=HOVER_HEIGHT_M * 1000.0,
                          min_above_table_mm=GRASP_MIN_ABOVE_TABLE_MM,
                          on_event=na_dogadjaj,
                          drop_yaw_deg=drop_yaw_robot_deg())
        # zona se trosi tek ovde  ako je pokret pukao ostaje slobodna za ponovni pokusaj
        zones.zauzmi()
        dbg(f"    hvat zavrsen. {zones.status()}")
    except Exception as e:
        demo(f"[!]        hvat nije uspio: {e} (zona {zone_no} ostaje slobodna)")
    demo("[sistem]   Cekam novu komandu.")


def _samo_hover(arm, x_mm, y_mm, rz_m, ryaw):
    #Rezervni potez kad hvat nije dozvoljen: samo dodji iznad predmeta.
    hz_mm = (rz_m + HOVER_HEIGHT_M) * 1000.0
    dbg(f"    -> umjesto hvata radim samo HOVER (z={hz_mm:.0f} mm).")
    try:
        arm.move_topdown(x_mm, y_mm, hz_mm, ryaw, speed=xk.SPEED)
    except Exception as e:
        demo(f"[!]        pokret nije uspio: {e}")
    demo("[sistem]   Cekam novu komandu.")


# GLAVNA PETLJA

def main():
    # Poruke po fazama
    dbg("1/5 Povezujem se na Pupil Capture...", flush=True)
    context, req, sub = pupilcore.setup_zmq()
    dbg("    OK - povezano.", flush=True)

    dbg("2/5 Ucitavam YOLO model...", flush=True)
    model = YOLO(YOLO_MODEL)
    dbg("    OK - YOLO spreman.", flush=True)

    tracker = dwell.DwellTracker()
    detector = aruco.setup_detector()
    marker_memory = aruco.MarkerMemory()          # zanemaruje kratke prekide detekcije
    # Skuplja poze predmeta u sistemu stola tokom dwell, pa vrati usrednjenu pozu
    pose_acc = aruco.TablePoseAccumulator(DWELL_THRESHOLD)

    dbg("3/5 Pravim board stola...", flush=True)
    table_board = aruco.make_table_board()       
    dbg("    OK - board spreman.", flush=True)

    dbg("4/5 Pripremam ruku...", flush=True)
    arm = setup_arm()                            

    zones = DropZones(DROP_TABLE_POINTS)

    dbg("5/5 Pokrecem petlju.", flush=True)
    demo(f"[sistem]   Spreman. Gledaj predmet {DWELL_THRESHOLD:.0f}s da ga izaberes. "
         f"('q' = izlaz, 'r' = oslobodi odlagalista)", flush=True)

    last_gaze = None   # zadnja pouzdana pozicija pogleda 
    last_frame_jpeg = None  # zadnjikadar iz scene kamere
    last_result = None   # zadnji YOLO rezultat 
    last_printed = None  # zadnje ispisano ime
    last_target_bbox = None   # zadnji okvir predmeta u koji gledamo
    frame_count = 0
    # normala stola iz proslog kadra njome se razrjesava dvoznacnost poze
    prev_table_normal = None

    last_action_ts = time.time()
    at_initial = True    # na startu ruka je u pocetnom polozaju

    najavljen = None   # predmet za koji je vec ispisano "Gledam u X"
    stol_prijavljen = False  # da li je ispisan pocetni spisak predmeta

    snimac = Snimac() if SNIMANJE else None
    arm_nit = None

    def ruka_zauzeta():
        return arm_nit is not None and arm_nit.is_alive()

    try:
        while True:
            #1. pokupi nove poruke sa mreze
            new_gaze, new_frame = pupilcore.drain_zmq_messages(sub)

            if new_gaze is not None:
                last_gaze = new_gaze
            if new_frame is not None:
                last_frame_jpeg = new_frame

            # nije stigao nov kadar nema sta da se crta
            if new_frame is None:
                cv2.waitKey(1)
                continue

            #2. dekodiraj sliku 
            frame = pupilcore.decode_jpeg(last_frame_jpeg)
            if frame is None:
                continue
            h, w = frame.shape[:2]

            #3. YOLO svaki Nti kadar
            frame_count += 1
            if frame_count % YOLO_EVERY_NTH_FRAME == 0:
                results = model(frame, verbose=False, imgsz=YOLO_IMG_SIZE)
                last_result = results[0]

            #4. ArUco svaki kadar
            now = time.time()
            markers_fresh = aruco.detect_and_estimate(frame, detector,
                                                      ref_normal=prev_table_normal)

            marker_memory.update(markers_fresh, now)
            markers = marker_memory.get_markers()
            table_pose = aruco.estimate_table_pose(markers_fresh, table_board)
            pose_acc.update(markers_fresh, table_pose, now)

            # zapamti normalu stola
            prev_table_normal = aruco.table_normal(table_pose)

            #5. crtanje
            if last_result is not None:
                detection.draw_yolo_boxes(frame, last_result, model.names)
            aruco.draw_detected(frame, markers)

            # sta se sve vidi na stolu
            if not stol_prijavljen and last_result is not None:
                nadjeni = []
                for i in range(len(last_result.boxes)):
                    if float(last_result.boxes.conf[i]) < YOLO_CONF_MIN:
                        continue
                    ime = model.names[int(last_result.boxes.cls[i])]
                    x1, y1, x2, y2 = last_result.boxes.xyxy[i].tolist()
                    m = detection.find_marker_in_bbox(
                        markers, (int(x1), int(y1), int(x2), int(y2)))
                    nadjeni.append(ime if m is not None
                                   else f"{ime} (BEZ markera - ne mogu ga uhvatiti)")
                if nadjeni:
                    crta()
                    demo(f"[stol]     Na stolu vidim: {', '.join(nadjeni)}.")
                    crta()
                    stol_prijavljen = True

            #6. u sta operator gleda 
            looked_at = None
            bbox = None
            if last_gaze is not None and last_result is not None:
                gx, gy = detection.gaze_to_pixel(last_gaze, w, h)

                # oznaka pogleda dva koncentricna kruga
                cv2.circle(frame, (gx, gy), GAZE_CIRCLE_RADIUS, GAZE_COLOR, 3)
                cv2.circle(frame, (gx, gy), 3, GAZE_COLOR, 1)

                # nadji predmet pod pogledom i istakni ga crvenim okvirom
                looked_at, bbox = detection.find_object_at_gaze(
                    last_result, gx, gy, model.names, conf_min=YOLO_CONF_MIN
                )
                if bbox is not None:
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                                  BBOX_HIGHLIGHT_COLOR, BBOX_HIGHLIGHT_THICKNESS)
                    last_target_bbox = bbox

            if looked_at != last_printed:
                dbg(f"Gledam: {looked_at if looked_at else '---'}")
                last_printed = looked_at

            # koliko dugo gledamo u isti predmet
            duration, just_selected = tracker.update(looked_at, now)

            if looked_at is None:
                najavljen = None
            elif looked_at != najavljen and duration >= DWELL_LOOK_ANNOUNCE_S:
                crta()
                demo(f"[pogled]   Gledam u {looked_at}.")
                najavljen = looked_at
            dwell.draw_dwell_progress(frame, tracker.current_target, duration,
                                      DWELL_THRESHOLD)

            #7. izbor potvrdjen salje se komanda ruci
            if just_selected:
                last_action_ts = now
                at_initial = False

                sel_bbox = bbox if bbox is not None else last_target_bbox
                #marker cije srediste pada unutar YOLO okvira pripada tom predmetu
                marker = detection.find_marker_in_bbox(markers, sel_bbox)
                if marker is None:
                    argumenti = (arm, zones, tracker.selected, None, None, 0, 0)
                else:
                    # uzmimamo usrednjenu pozu skupljanu tokom dwell-a
                    T_avg, n_in, n_tot = pose_acc.averaged(marker["id"], now)
                    argumenti = (arm, zones, tracker.selected, marker["id"],
                                 T_avg, n_in, n_tot)

                if not ASINHRONA_RUKA or arm is None:
                    send_to_xarm(*argumenti)
                elif ruka_zauzeta():
                    demo("[sistem]   Ruka je jos zauzeta - sacekaj da zavrsi pa "
                         "pogledaj ponovo.")
                else:
                    arm_nit = threading.Thread(target=send_to_xarm, args=argumenti,
                                               daemon=True)
                    arm_nit.start()

            #povratak u pocetni polozaj poslije mirovanja
            if (arm is not None and not at_initial and not ruka_zauzeta()
                    and (now - last_action_ts) >= IDLE_RETURN_S):
                crta()
                demo(f"[sistem]   Proslo je {IDLE_RETURN_S:.0f}s od poslednje "
                     f"aktivacije, ruka se vraca u inicijalnu poziciju.")
                try:
                    arm.go_initial()
                except Exception as e:
                    demo(f"[!]        povratak nije uspio: {e}")
                at_initial = True   

            # koliko je predmeta odlozeno i koja je zona sljedeca
            cv2.putText(frame, zones.status(), (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, GAZE_COLOR, 2)
            nacrtaj_log(frame)

            cv2.imshow("Pupil core - world + gaze", frame)
            if snimac is not None:
                snimac.dodaj(frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('r'):
                #novi snimak demoa 
                zones.reset()
                crta()
                demo(f"[sistem]   Odlagalista oslobodjena ({zones.status()}).")

    except KeyboardInterrupt:
        demo("\nZaustavljam..")
    finally:
        if snimac is not None:
            snimac.zatvori()
        if arm is not None:
            arm.disconnect()
        cv2.destroyAllWindows()
        sub.close()
        req.close()
        context.term()
        demo("Zatvoreno.")


if __name__ == "__main__":
    main()
