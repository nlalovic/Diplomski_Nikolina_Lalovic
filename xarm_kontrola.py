from config import (
    LIFT_AFTER_GRASP_MM,
    RELEASE_ABOVE_PICK_MM,
    RETREAT_AFTER_RELEASE_MM,
    CARRY_MIN_ABOVE_TABLE_MM,
)

XARM_IP = "10.1.108.143" # IP adresa xArm kontrolera

SPEED = 60 # mm/s spori precizni pokreti oko predmeta spustanje, hvat, pustanje
SPEED_TRAVEL = 120 # mm/s brzi pomjeraji gdje nema opasnosti od sudara

DOWN_ROLL = 180.0 # top-down hvat
DOWN_PITCH = 0.0 # Yaw se ne fiksira, dolazi iz vizije

HOVER_ABOVE_REF_MM = 50.0
Z_MIN = -200.0

USE_GRIPPER = True

GRIPPER_SPEED = 2000 # brzina otvaranja/zatvaranja u jedinicama SDK

GRIPPER_POS_OPEN = 850 # pozicija 850 prsti otvoreni 85 mm
GRIPPER_OPEN_MM = 85.0
GRIPPER_POS_CLOSED = -10 # pozicija -10 = 0 mm.
GRIPPER_CLOSED_MM = 0.0
GRIP_MARGIN_MM = 8.0


class XArmController:

    def __init__(self, ip=XARM_IP, quiet=False):
        self.ip = ip
        self.arm = None
        self.quiet = quiet
        self.initial_pose = None

    def _say(self, msg):
        if not self.quiet:
            print(msg)

    # zivotni ciklus konekcije
    def connect(self):
        
        from xarm.wrapper import XArmAPI

        if self.quiet:
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()):
                self._open_and_arm(XArmAPI)
        else:
            self._open_and_arm(XArmAPI)

        self.initial_pose = self._read_pose()
        self._say(f"[xArm] povezan na {self.ip}, spreman.")

    def _open_and_arm(self, XArmAPI):
        self.arm = XArmAPI(self.ip)
        self.arm.clean_warn() # ocisti stare waringe i errorer da ne blokiraju pokret
        self.arm.clean_error()
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)  # 0 = position control
        self.arm.set_state(0) # 0 = ready
        if USE_GRIPPER:
            self.arm.set_gripper_enable(True)
            self.arm.set_gripper_mode(0)
            self.arm.set_gripper_speed(GRIPPER_SPEED)
        else:
            self._say("[xArm] USE_GRIPPER=False -> preskacem hvataljku.")

    def _read_pose(self):
        try:
            ret = self.arm.get_position()
            if isinstance(ret, (list, tuple)) and len(ret) == 2 and ret[0] == 0:
                return tuple(float(v) for v in ret[1][:6])
        except Exception as e:
            print(f"[xArm] (ne mogu procitati poziciju: {e})")
        return None

    def disconnect(self):
        if self.arm is not None:
            self.arm.disconnect()
            self.arm = None
            self._say("[xArm] konekcija zatvorena.")

    # osnovni pokreti 
    def _check(self, code, sta):
        if code != 0:
            try:
                ret = self.arm.get_err_warn_code()
                err_warn = ret[1] if isinstance(ret, (list, tuple)) and len(ret) > 1 else ret
                print(f"[xArm] DIJAGNOSTIKA: err/warn kod = {err_warn}  "
                      f"(err!=0 -> greska, warn!=0 -> upozorenje; 0,0 = cisto)")
            except Exception as e:
                print(f"[xArm] (ne mogu procitati err/warn kod: {e})")
            raise RuntimeError(f"[xArm] {sta} nije uspio (code={code}).")

    def move_topdown(self, x_mm, y_mm, z_mm, yaw_deg, speed=SPEED):
        z_mm = max(z_mm, Z_MIN)    # sigurnosna granica
        code = self.arm.set_position(
            x=x_mm, y=y_mm, z=z_mm,
            roll=DOWN_ROLL, pitch=DOWN_PITCH, yaw=yaw_deg,
            speed=speed, wait=True,
        )
        self._check(code, f"move na ({x_mm:.0f},{y_mm:.0f},{z_mm:.0f})")

    def go_initial(self, speed=SPEED_TRAVEL):
        x, y, z, r, p, yw = self.initial_pose
        code = self.arm.set_position(x=x, y=y, z=max(z, Z_MIN), roll=r, pitch=p,
                                     yaw=yw, speed=speed, wait=True)
        self._check(code, "povratak u pocetni polozaj")
        self._say("[xArm] ruka vracena u pocetni polozaj.")
        return True

    def _mm_to_units(self, mm):
        # Zeljeni razmak prstiju (mm) u jedinice SDK-a
        frac = (mm - GRIPPER_CLOSED_MM) / (GRIPPER_OPEN_MM - GRIPPER_CLOSED_MM)
        units = GRIPPER_POS_CLOSED + frac * (GRIPPER_POS_OPEN - GRIPPER_POS_CLOSED)
        units = max(GRIPPER_POS_CLOSED, min(GRIPPER_POS_OPEN, units))
        return int(round(units))

    def set_gripper_mm(self, mm):
        # Postavi prste na zadati razmak u mm
        units = self._mm_to_units(mm)
        code = self.arm.set_gripper_position(units, wait=True)
        self._check(code, f"gripper na {mm:.0f} mm ({units} jed.)")

    # pun ciklus
    def pick_top_down(self, x_mm, y_mm, grasp_z_mm, yaw_deg, width_mm, height_mm,
                      drop_xyz, close_mm=None,
                      hover_mm=HOVER_ABOVE_REF_MM, min_above_table_mm=0.0,
                      on_event=None, drop_yaw_deg=None):
        def ev(naziv, z_mm):
            if on_event is not None:
                on_event(naziv, z_mm)
        open_mm = GRIPPER_OPEN_MM
        if close_mm is None:
            close_mm = max(0.0, width_mm - GRIP_MARGIN_MM)

        # Ravan stola u k.s. robota
        dx, dy, table_z = drop_xyz

        grasp_z = grasp_z_mm                        # (2) tu se stisne
        hover_z = grasp_z + hover_mm                # (1) prilaz odozgo
        lift_z = max(grasp_z + LIFT_AFTER_GRASP_MM,
                     table_z + CARRY_MIN_ABOVE_TABLE_MM)
        release_z = grasp_z + RELEASE_ABOVE_PICK_MM        # (6) pustanje
        retreat_z = release_z + RETREAT_AFTER_RELEASE_MM   # (8) povlacenje

        # SIGURNOSNA PROVJERA PRIJE IJEDNOG POKRETA
        grasp_above_table = grasp_z - table_z
        if grasp_above_table < min_above_table_mm:
            raise RuntimeError(
                f"tacka hvata bi bila {grasp_above_table:.0f} mm iznad stola "
                f"(dozvoljeno najmanje {min_above_table_mm:.0f} mm). "
                f"grasp_z={grasp_z:.0f} sto={table_z:.0f} mm. "
                f"NE hvatam"
            )

        self._say(f"[xArm] HVAT: predmet sirina {width_mm:.0f} visina {height_mm:.0f} mm | "
                  f"griper otvor MAX {open_mm:.0f} mm, stisak {close_mm:.0f} mm")
        if drop_yaw_deg is not None:
            self._say(f"       hvat pod yaw={yaw_deg:+.1f}, odlaganje pod "
                      f"yaw={drop_yaw_deg:+.1f} deg")
        self._say(f"       prilaz {hover_z:.0f} | hvat {grasp_z:.0f} "
                  f"({grasp_above_table:.0f} mm iznad stola) | nosi {lift_z:.0f} | "
                  f"pusta {release_z:.0f} | povlaci {retreat_z:.0f} mm")

        self.set_gripper_mm(open_mm)                                       # (0)
        self.move_topdown(x_mm, y_mm, hover_z, yaw_deg, speed=SPEED_TRAVEL)  # (1)
        self.move_topdown(x_mm, y_mm, grasp_z, yaw_deg, speed=SPEED)         # (2)
        self.set_gripper_mm(close_mm)                                        # (3)
        ev("uhvacen", grasp_z)
        self.move_topdown(x_mm, y_mm, lift_z, yaw_deg, speed=SPEED)          # (4)
        ev("prenosim", lift_z)

        # (5) Prenos na odlagaliste
        odlaganje_yaw = yaw_deg if drop_yaw_deg is None else drop_yaw_deg
        self.move_topdown(dx, dy, lift_z, odlaganje_yaw, speed=SPEED_TRAVEL)
        self.move_topdown(dx, dy, release_z, odlaganje_yaw, speed=SPEED)     # (6)
        self.set_gripper_mm(open_mm)                                         # (7)
        ev("spusten", release_z)
        self.move_topdown(dx, dy, retreat_z, odlaganje_yaw, speed=SPEED_TRAVEL)  # (8)
        self._say("[xArm] HVAT zavrsen.")


if __name__ == "__main__":
    # DRY test
    print("Dry test konekcije. Trenutni IP:", XARM_IP)
    a = XArmController()
    a.connect()
    print("  trenutna pozicija:", a.arm.get_position())
    a.disconnect()
