"""DWELL - izbor predmeta zadrzavanjem pogleda.

Ovo je mehanizam kojim operator "pritiska dugme" ocima. Nema klika ni glasovne
komande: ako pogled ostane na istom predmetu duze od DWELL_THRESHOLD sekundi,
predmet se smatra izabranim i ciklus hvatanja krece.

ZASTO GRACE PERIOD: oko nije mirno ni kad "gledas u jednu tacku" - trepces
(200-300 ms) i pravis sitne sakade. Bez tolerancije bi svaki treptaj ponistio
napredak i izbor prakticno ne bi bio moguc. Zato kratak prekid ne resetuje
brojanje.
"""

import cv2

from config import (
    DWELL_THRESHOLD, DWELL_GRACE_PERIOD,
    DWELL_BAR_W, DWELL_BAR_H, DWELL_BAR_X, DWELL_BAR_Y,
)


class DwellTracker:
    #Prati koliko dugo operator gleda u isti predmet

    def __init__(self, threshold=DWELL_THRESHOLD, grace_period=DWELL_GRACE_PERIOD):
        self.threshold = threshold # prag zadrzavanja
        self.grace_period = grace_period # tolerancija (zbog treptaja, pomjeraja)
        self.current_target = None   # klasa predmeta koji trenutno gledamo
        self.dwell_start = None  # kad je gledanje pocelo
        self.last_seen_time = None # kad je pogled zadnji put bio na predmetu
        self.selected = None  # predmet koji je presao prag

    def update(self, looked_at, now): # vraca (koliko dugo gledamo predmet, da li je presao prag)
        # SLUCAJ 1: gledamo isti predmet kao i prije
        if looked_at == self.current_target and self.current_target is not None:
            self.last_seen_time = now
            duration = now - self.dwell_start

            just_selected = False
            if duration >= self.threshold and self.selected != self.current_target:
                self.selected = self.current_target
                just_selected = True
                print(f"+ SELECTED: {self.selected}")

            return duration, just_selected

        # SLUCAJ 2: pogled je presao na drugi predmet - brojanje krece ispocetka
        if looked_at is not None:
            self.current_target = looked_at
            self.dwell_start = now
            self.last_seen_time = now
            self.selected = None
            return 0.0, False

        # SLUCAJ 3: nema pogleda - ako je prekid kratak (treptaj) nastavljamo brojanje
        if self.last_seen_time is not None:
            if now - self.last_seen_time <= self.grace_period:
                return now - self.dwell_start, False

        # prekid je predug - korisnik je stvarno skrenuo pogled, resetuj sve
        self.current_target = None
        self.dwell_start = None
        self.last_seen_time = None
        self.selected = None
        return 0.0, False


def draw_dwell_progress(frame, target_name, duration, threshold): # crtanje progress bar-a za dwell
    if target_name is None or duration <= 0:
        return

    progress = min(duration / threshold, 1.0)
    fill_w = int(DWELL_BAR_W * progress)
    bar_color = (0, 255, 0) if progress >= 1.0 else (0, 165, 255)

    # pozadina trake
    cv2.rectangle(frame,
                  (DWELL_BAR_X, DWELL_BAR_Y),
                  (DWELL_BAR_X + DWELL_BAR_W, DWELL_BAR_Y + DWELL_BAR_H),
                  (50, 50, 50), -1)
    # popunjeni dio
    cv2.rectangle(frame,
                  (DWELL_BAR_X, DWELL_BAR_Y),
                  (DWELL_BAR_X + fill_w, DWELL_BAR_Y + DWELL_BAR_H),
                  bar_color, -1)
    # okvir
    cv2.rectangle(frame,
                  (DWELL_BAR_X, DWELL_BAR_Y),
                  (DWELL_BAR_X + DWELL_BAR_W, DWELL_BAR_Y + DWELL_BAR_H),
                  (255, 255, 255), 1)
    # tekst ispod trake
    label = f"{target_name}: {duration:.1f}s / {threshold:.1f}s"
    cv2.putText(frame, label,
                (DWELL_BAR_X, DWELL_BAR_Y + DWELL_BAR_H + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
