import zmq
import msgpack
import cv2
import numpy as np

from config import PUPIL_IP, PUPIL_REQ_PORT, GAZE_CONFIDENCE_MIN


def setup_zmq():
    context = zmq.Context()
    # prvo trazimo port na kom se objavljuju podaci
    req = zmq.Socket(context, zmq.REQ)
    req.setsockopt(zmq.RCVTIMEO, 5000) # timeout 5s, da program ne ostane u beskonacnoj petlji
    req.setsockopt(zmq.LINGER, 0)
    req.connect(f"tcp://{PUPIL_IP}:{PUPIL_REQ_PORT}")
    req.send_string("SUB_PORT")
    try:
        sub_port = req.recv_string()
    except zmq.Again:
        raise RuntimeError(
            f"Pupil Capture ne odgovara na {PUPIL_IP}:{PUPIL_REQ_PORT} u roku od 5s. "
            f"Provjeri: Pupil Capture upaljen + Network API aktivan na portu "
            f"{PUPIL_REQ_PORT}."
        )
    # zatim se pretplacujemo na port
    sub = zmq.Socket(context, zmq.SUB)
    sub.connect(f"tcp://{PUPIL_IP}:{sub_port}")
    sub.subscribe("gaze")  # podaci o pogledu
    sub.subscribe("frame.world")  # slika scene kamere 30 fps

    return context, req, sub


def drain_zmq_messages(sub):
    new_gaze = None
    new_frame_jpeg = None

    while True:
        try:
            topic = sub.recv_string(flags=zmq.NOBLOCK)  # tema
            payload = sub.recv()   # msgpack (norm_pos,confidence za topic gaze)
            msg = msgpack.unpackb(payload)

            if topic.startswith("gaze"):
                if (msg.get("confidence") or 0) > GAZE_CONFIDENCE_MIN:
                    new_gaze = msg.get("norm_pos")

            elif topic.startswith("frame.world"):
                new_frame_jpeg = sub.recv() # slika dolazi kao treci dio poruke na topic frame.world

        except zmq.Again:
            break     # bafer je prazan

    return new_gaze, new_frame_jpeg


def decode_jpeg(jpeg_bytes):
    jpeg_array = np.frombuffer(jpeg_bytes, dtype=np.uint8) # memorijski bafer pretvaramo u numpy
    return cv2.imdecode(jpeg_array, cv2.IMREAD_COLOR) # dekodovanje slike, vraca siirnu, visinu i BGR
