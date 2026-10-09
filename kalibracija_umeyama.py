"""Kalibracija STO -> ROBOT preko KABSCH-UMEYAMA (rigid registration).

CILJ: naci 4x4 matricu T_ROBOT_TABLE koja pretvara poziciju izrazenu u sistemu
STOLA (kako je kamera racuna preko ceoskastih markera 21-24) u sistem BAZE ROBOTA
(u kome xArm prima komande). Sto i robot su fiksni -> matrica je KONSTANTA, mjeri
se jednom.

METOD MJERENJA (nacrtane tacke, laboratorija 2026-07-10):
  Robot ne dohvata sva 4 ceoska, pa se NE oslanjamo na dodir ceoskastih markera.
  Umjesto toga koristimo LISTU POZNATIH TACAKA (KALIB_TACKE dolje) koje robot MOZE
  da dohvati i kojima znamo sto-koordinate (nacrtane/izmjerene lenjirom od centra
  markera 24: X duz pravca 24->23, Y duz pravca 24->21). Vrhom (kalibrisani TCP)
  dodirnes svaku tacku, procitas poziciju iz robota (get_position), i Kabsch-Umeyama
  nadje najbolju (least-squares) transformaciju sto -> robot.

KOORDINATNI SISTEM STOLA (mora se poklapati sa onim koji kamera koristi!):
  - ishodiste (0,0) = centar markera 24 (dole-lijevo)
  - X osa = pravac 24 -> 23 (donja ivica), raste do ~0.804 m
  - Y osa = pravac 24 -> 21 (lijeva ivica), raste do ~0.604 m
  - Z = visina IZNAD ravni stola (0 za tacke na stolu)

PLANARNI MOD (FORCE_PLANAR=True, nas slucaj):
  Z osa robota je POTVRDJENO paralelna Z osi stola. Zato Umeyama racunamo SAMO u
  ravni (X,Y): trazimo yaw (zaokret oko vertikale) + XY pomak, a nagib (roll,pitch)
  je NAMETNUT na 0. Tako se sum iz Z ocitanja NE pretvara u laznu rotaciju. Visinu
  stola Z0 vadimo posebno (prosjek robotZ umanjen za poznatu visinu svake tacke).

KABSCH-UMEYAMA: opsta metoda rigid poravnanja dva skupa tacaka (Umeyama 1991,
"Least-squares estimation of transformation parameters between two point patterns").
Funkcija `kabsch_umeyama` radi i 2D i 3D, korektno rjesava refleksiju (matrica S),
i moze da procijeni SKALU (kod nas samo dijagnostika - mora ispasti ~1.0).

TCP: kalibrisan, payload 0, offset samo po Z osi. Zato get_position vraca sam VRH,
i nagib alata ne kvari X,Y (robot preracuna vrh za bilo koju orijentaciju). Uspravno
je i dalje najsigurnije, ali nije uslov.

BEZBJEDNOST: skript SAMO CITA poziciju (get_position). NIKAD ne salje pokret.
Ruku pomjeras rukom (free-drive) ili jogom iz UFactory Studija.
"""

import json
import numpy as np

# =========================================================================
# PODESAVANJA
# =========================================================================
ROBOT_IP = "10.1.108.143"      # xArm u laboratoriji

# LISTA KALIBRACIONIH TACAKA: (citljivo_ime, (x, y, z)) u METRIMA, u sistemu STOLA
KALIB_TACKE = [
    ("tacka_1",        (0.271, 0.000, 0.000)),   # nacrtana, na stolu
    ("tacka_2",        (0.044, 0.400, 0.000)),   # nacrtana, na stolu
    ("marker_23",      (0.804, 0.000, 0.003)),   # centar markera 23 je 3 mm IZNAD stola
    ("ispod_22_15cm",  (0.804, 0.453, 0.000)),   # 15.1 cm ispod centra markera 22 (ka 23)
]

USE_FREEDRIVE = False
FORCE_PLANAR = True


# MATEMATIKA: KABSCH-UMEYAMA (rigid, radi 2D i 3D, opciono sa skalom)
def kabsch_umeyama(src, dst, with_scale=False):
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    assert src.shape == dst.shape, "src i dst moraju biti istih dimenzija (N,d)"
    n, d = src.shape

    # 1) centroidi i centrirane tacke
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    # 2) kros-kovarijansa (d x d)
    H = (dst_c.T @ src_c) / n

    # 3) SVD
    U, D, Vt = np.linalg.svd(H)

    # 4) refleksiona popravka (Umeyama)
    S = np.eye(d)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0.0:
        S[-1, -1] = -1.0

    # 5) rotacija
    R = U @ S @ Vt

    # 6) skala
    if with_scale:
        var_src = (src_c ** 2).sum() / n          
        s = float((D * np.diag(S)).sum() / var_src)
    else:
        s = 1.0

    # 7) translacija
    t = mu_dst - s * R @ mu_src

    T = np.eye(d + 1, dtype=np.float64)
    T[:d, :d] = s * R
    T[:d, d] = t
    return R, t, s, T


def rms_error_mm(T, src_xyz, dst_xyz):
    src = np.asarray(src_xyz, dtype=np.float64)
    dst = np.asarray(dst_xyz, dtype=np.float64)
    src_h = np.hstack([src, np.ones((src.shape[0], 1))])   
    pred = (T @ src_h.T).T[:, :3]                           
    errs = np.linalg.norm(pred - dst, axis=1)              
    rms = float(np.sqrt((errs ** 2).mean())) * 1000.0
    return rms, errs * 1000.0


def rotation_info(R):
    yaw = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    cos_tilt = max(-1.0, min(1.0, float(R[2, 2])))
    tilt = float(np.degrees(np.arccos(cos_tilt)))
    return yaw, tilt


# GLAVNI TOK

def _collect_point(arm, ime, src_xyz):
    cx, cy, cz = src_xyz
    while True:
        ans = input(f"\n-> {ime}  (sto: x={cx:.3f} y={cy:.3f} z={cz:.3f} m). "
                    f"Vrh na tacku, pa Enter: ").strip().lower()
        if ans == "s":
            print("   preskoceno.")
            return None

        code, pose = arm.get_position(is_radian=False)   # [x,y,z,roll,pitch,yaw], mm/deg
        if code != 0:
            print(f"   GRESKA citanja pozicije (code={code}), pokusaj ponovo.")
            continue

        x_m, y_m, z_m = pose[0] / 1000.0, pose[1] / 1000.0, pose[2] / 1000.0
        print(f"   procitano (robot): X={x_m:+.4f} Y={y_m:+.4f} Z={z_m:+.4f} m")
        return np.array([cx, cy, cz]), np.array([x_m, y_m, z_m])


def main():
    from xarm.wrapper import XArmAPI   # import ovdje da skript moze da se ucita i bez SDK-a

    print(f"Povezujem se na xArm ({ROBOT_IP})...", flush=True)
    arm = XArmAPI(ROBOT_IP)
    arm.motion_enable(True)
    arm.set_state(0)

    if USE_FREEDRIVE:
        arm.set_mode(2)   
        arm.set_state(0)
        print("MANUAL mod ukljucen - ruku mozes pomjerati rukom.\n")
    else:
        print("Ruku pomjeraj jogom.\n")

    print("=" * 64)
    print("KALIBRACIJA STO -> ROBOT  (Umeyama)")
    print("Za svaku tacku: dovedi vrh alata tacno na nju, pa pritisni ENTER.")
    print("('s' + Enter = preskoci tu tacku)")
    print("=" * 64)

    src_pts = []   # sto-koordinate (m), 3D
    dst_pts = []   # izmjerene robot-koordinate (m), 3D

    for ime, xyz in KALIB_TACKE:
        got = _collect_point(arm, ime, xyz)
        if got is not None:
            src_pts.append(got[0])
            dst_pts.append(got[1])

    n = len(src_pts)
    if n < 3:
        print(f"\nPremalo tacaka ({n}). Treba bar 3 (ne kolinearne). Prekidam.")
        return

    src = np.array(src_pts)
    dst = np.array(dst_pts)

    if FORCE_PLANAR:
        # kalibracija u ravni (X,Y): yaw + XY pomak, nagib nametnut na 0
        R2, t2, _, _ = kabsch_umeyama(src[:, :2], dst[:, :2], with_scale=False)
        z0 = float((dst[:, 2] - src[:, 2]).mean())
        yaw = float(np.degrees(np.arctan2(R2[1, 0], R2[0, 0])))
        tilt = 0.0                                    
        T = np.eye(4, dtype=np.float64)
        T[:2, :2] = R2
        T[:3, 3] = [t2[0], t2[1], z0]
        t = T[:3, 3]
        _, _, s_diag, _ = kabsch_umeyama(src[:, :2], dst[:, :2], with_scale=True)
    else:
        R, t, _, T = kabsch_umeyama(src, dst, with_scale=False)
        yaw, tilt = rotation_info(R)
        _, _, s_diag, _ = kabsch_umeyama(src, dst, with_scale=True)

    rms, per_point = rms_error_mm(T, src, dst)

    # =====================================================================
    # ISPIS
    # =====================================================================
    print("\n" + "=" * 64)
    print("REZULTAT  (Kabsch-Umeyama)")
    print("=" * 64)
    print(f"Broj tacaka            : {n}")
    print(f"Yaw (sto->robot)       : {yaw:+.2f} deg")
    print(f"Nagib ravni stola      : {tilt:.2f} deg  (prema robotovoj vertikali)")
    print(f"Translacija X0,Y0,Z0   : {t[0]:+.4f}, {t[1]:+.4f}, {t[2]:+.4f} m")
    print(f"Skala (dijagnostika)   : {s_diag:.5f}  (treba ~1.000; odstupanje = "
          f"greska u mjerenju tacaka ili pomijesane jedinice)")
    print(f"\nRMS greska             : {rms:.2f} mm")
    print("Greska po tacki (mm)   : " + ", ".join(f"{e:.1f}" for e in per_point))

    # snimi u json
    out = {
        "metod": "umeyama_planar" if FORCE_PLANAR else "umeyama_3d",
        "yaw_deg": yaw,
        "tilt_deg": tilt,
        "t_xyz_m": [float(v) for v in t],
        "skala_dijagnostika": s_diag,
        "rms_mm": rms,
        "n_points": n,
        "imena_tacaka": [ime for ime, _ in KALIB_TACKE][:n],
        "src_table_xyz_m": src.tolist(),
        "dst_robot_xyz_m": dst.tolist(),
        "T_ROBOT_TABLE": T.tolist(),
    }
    with open("robot_table_kalibracija_umeyama.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nSnimljeno u robot_table_kalibracija_umeyama.json")


if __name__ == "__main__":
    main()
