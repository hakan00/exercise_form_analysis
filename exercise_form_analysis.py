import cv2
import mediapipe as mp
import numpy as np
from collections import deque

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
PL = mp_pose.PoseLandmark


def angle3(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ang = np.abs((np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])) * 180 / np.pi)
    return 360 - ang if ang > 180 else ang

def vert_angle(a, b):
    return np.abs(np.arctan2(b[0]-a[0], -(b[1]-a[1])) * 180 / np.pi)

def pt(landmarks, idx):
    p = landmarks.landmark[idx.value]
    return [p.x, p.y]


class BaseAnalyzer:
    EXERCISE_NAME = ""

    def __init__(self):
        self.counter = 0
        self.good_reps = 0
        self.stage = None
        self.form_scores = deque(maxlen=100)
        self.feedback = []
        self.rep_log = []
        self._bufs = {}
        self._last_ang = {}
        self._rep_sum = 0
        self._rep_n = 0
        self._rep_errs = []

    def _smooth(self, key, val):
        if key not in self._bufs:
            self._bufs[key] = deque(maxlen=7)
        self._bufs[key].append(val)
        v = float(np.mean(self._bufs[key]))
        self._last_ang[key] = v
        return v

    def _finish_rep(self):
        score = int(self._rep_sum / max(1, self._rep_n))
        errs = list(set(self._rep_errs))
        if score >= 60:
            self.good_reps += 1
        self.counter += 1
        self.rep_log.append({"rep": self.counter, "score": score, "errors": errs})
        self._rep_sum = 0
        self._rep_n = 0
        self._rep_errs = []
        return score, errs, score >= 60

    def get_summary(self):
        if not self.rep_log:
            return None
        scores = [r["score"] for r in self.rep_log]
        return {
            "total": self.counter,
            "good": self.good_reps,
            "avg_score": float(np.mean(scores)),
            "reps": self.rep_log
        }

    def _draw_header(self, frame, r):
        h, w = frame.shape[:2]
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (w, 80), (30, 30, 30), -1)
        cv2.rectangle(ov, (0, 80), (220, h-100), (30, 30, 30), -1)
        cv2.rectangle(ov, (0, h-100), (w, h), (30, 30, 30), -1)
        cv2.addWeighted(ov, 0.7, frame, 0.3, 0, frame)

        cv2.putText(frame, self.EXERCISE_NAME, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(frame, f"Reps: {r['counter']}", (15, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        st = "UP" if r["stage"] == "up" else "DOWN" if r["stage"] == "down" else "READY"
        sc = (0,255,0) if r["stage"] == "up" else (0,165,255)
        cv2.putText(frame, st, (185, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, sc, 2)
        cv2.putText(frame, f"Good: {r['good_reps']}", (305, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        s = r["form_score"]
        bx = w - 210
        clr = (0,255,0) if s >= 80 else (0,200,255) if s >= 50 else (0,0,255)
        cv2.rectangle(frame, (bx, 25), (bx+190, 42), (60,60,60), -1)
        cv2.rectangle(frame, (bx, 25), (bx+int(190*s/100), 42), clr, -1)
        cv2.putText(frame, f"Form: {int(s)}%", (bx, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        if r["feedback"]:
            cv2.putText(frame, r["feedback"][0][0], (15, h-25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, r["feedback"][0][1], 2)


class SquatAnalyzer(BaseAnalyzer):
    EXERCISE_NAME = "BARBELL SQUAT"

    def __init__(self):
        super().__init__()
        self.KNEE_DOWN = 105
        self.KNEE_UP = 165
        self.BACK_THRESH = 40
        self.DEPTH_THRESH = 90
        self._min_knee = 180
        self._sticky_msg = ("Perfect form!", (0,255,0))
        self._sticky_frames = 0

    def analyze_frame(self, landmarks):
        self.feedback = []
        fs = 100

        lh = pt(landmarks, PL.LEFT_HIP)
        rh = pt(landmarks, PL.RIGHT_HIP)
        lk = pt(landmarks, PL.LEFT_KNEE)
        rk = pt(landmarks, PL.RIGHT_KNEE)
        la = pt(landmarks, PL.LEFT_ANKLE)
        ra = pt(landmarks, PL.RIGHT_ANKLE)
        ls = pt(landmarks, PL.LEFT_SHOULDER)
        rs = pt(landmarks, PL.RIGHT_SHOULDER)

        lka = self._smooth("lk", angle3(lh, lk, la))
        rka = self._smooth("rk", angle3(rh, rk, ra))
        avg_knee = (lka + rka) / 2

        back_angle = vert_angle(
            [(lh[0]+rh[0])/2, (lh[1]+rh[1])/2],
            [(ls[0]+rs[0])/2, (ls[1]+rs[1])/2]
        )
        back_ok = back_angle < self.BACK_THRESH
        if not back_ok:
            fs -= 25
            self.feedback.append(("EXCESSIVE FORWARD LEAN!", (0,0,255)))
            self._rep_errs.append("back_lean")

        knee_d = abs(lk[0]-rk[0])
        hip_d = abs(lh[0]-rh[0])
        valgus_ok = not (hip_d > 0.01 and knee_d < hip_d*0.60 and self.stage == "down")
        if not valgus_ok:
            fs -= 20
            self.feedback.append(("Knees caving in! Push them outward.", (0,0,255)))
            self._rep_errs.append("knee_valgus")

        self._min_knee = min(self._min_knee, min(lka, rka))
        self._rep_sum += max(0, fs)
        self._rep_n += 1

        if avg_knee < self.KNEE_DOWN:
            self.stage = "down"

        if avg_knee > self.KNEE_UP and self.stage == "down":
            self.stage = "up"
            if self._min_knee > self.DEPTH_THRESH:
                self.feedback.append(("PARTIAL SQUAT! Go deeper.", (0,165,255)))
                self._rep_sum = 0
                self._rep_n = 0
                self._rep_errs = []
            else:
                self._finish_rep()
            self._min_knee = 180

        if not self.feedback:
            self.feedback.append(("Perfect form!", (0,255,0)))

        if self.feedback[0][0] != "Perfect form!":
            self._sticky_msg = self.feedback[0]
            self._sticky_frames = 60
        elif self._sticky_frames > 0:
            self.feedback = [self._sticky_msg]
            self._sticky_frames -= 1

        fs = max(0, min(100, fs))
        self.form_scores.append(fs)
        return {
            "left_knee": lka, "right_knee": rka, "avg_knee": avg_knee,
            "back_angle": back_angle, "back_ok": back_ok, "valgus_ok": valgus_ok,
            "knee_d": knee_d, "hip_d": hip_d,
            "form_score": fs, "counter": self.counter, "good_reps": self.good_reps,
            "stage": self.stage, "feedback": self.feedback
        }

    def draw_info_panel(self, frame, r):
        self._draw_header(frame, r)
        rows = [
            (f"L Knee:  {r['left_knee']:.0f}",  105),
            (f"R Knee:  {r['right_knee']:.0f}",  130),
            (f"Avg:     {r['avg_knee']:.0f}",    155),
            (f"Back:    {r['back_angle']:.1f}",  180),
            (f"KneeW:  {r['knee_d']:.3f}",       205),
            (f"HipW:   {r['hip_d']:.3f}",        225),
        ]
        for txt, y in rows:
            cv2.putText(frame, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)

        for lbl, ok, y in [("Back", r["back_ok"], 250), ("Knees", r["valgus_ok"], 275)]:
            c = (0,255,0) if ok else (0,0,255)
            cv2.circle(frame, (20, y-5), 7, c, -1)
            cv2.putText(frame, f"{'OK' if ok else '!!'} {lbl}", (35, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)

    def draw_angles(self, frame, landmarks):
        lml = landmarks.landmark
        h, w = frame.shape[:2]
        for key, idx in [("lk", PL.LEFT_KNEE), ("rk", PL.RIGHT_KNEE)]:
            p = lml[idx.value]
            cv2.putText(frame, f"{int(self._last_ang.get(key, 0))}",
                        (int(p.x*w)-15, int(p.y*h)-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)


class ShoulderPressAnalyzer(BaseAnalyzer):
    EXERCISE_NAME = "SHOULDER PRESS"

    def __init__(self):
        super().__init__()

        self.ELBOW_DOWN = 95
        self.ELBOW_UP   = 150

        self.ARM_POS_MIN  = 70
        self.ARM_POS_MAX  = 110
        self.WRIST_THRESH = 5
        self.SYM_THRESH   = 10
        self.ELBOW_MIN    = 45

        self._min_elbow = 180
        self._max_elbow = 0
        self._depth_ok  = True
        self._last_rep  = None
        self._rep_show  = 0

    def analyze_frame(self, landmarks):
        self.feedback = []
        fs = 100

        ls = pt(landmarks, PL.LEFT_SHOULDER)
        le = pt(landmarks, PL.LEFT_ELBOW)
        lw = pt(landmarks, PL.LEFT_WRIST)
        rs = pt(landmarks, PL.RIGHT_SHOULDER)
        re = pt(landmarks, PL.RIGHT_ELBOW)
        rw = pt(landmarks, PL.RIGHT_WRIST)
        lh = pt(landmarks, PL.LEFT_HIP)
        rh = pt(landmarks, PL.RIGHT_HIP)

        lea = self._smooth("le", angle3(ls, le, lw))
        rea = self._smooth("re", angle3(rs, re, rw))
        avg_elbow = (lea + rea) / 2

        laa = self._smooth("la", angle3(lh, ls, le))
        raa = self._smooth("ra", angle3(rh, rs, re))
        avg_arm = (laa + raa) / 2

        lw_ang = self._smooth("lw", vert_angle(le, lw))
        rw_ang = self._smooth("rw", vert_angle(re, rw))
        avg_wrist = (lw_ang + rw_ang) / 2

        ediff = abs(lea - rea)

        self._min_elbow = min(self._min_elbow, avg_elbow)
        self._max_elbow = max(self._max_elbow, avg_elbow)
        if avg_elbow < self.ELBOW_DOWN:
            self._depth_ok = True

        elbow_ok = self._depth_ok
        if avg_elbow < self.ELBOW_MIN:
            elbow_ok = False
            fs -= 20
            self.feedback.append(("Too deep! Don't over-bend elbows.", (0,0,255)))
            self._rep_errs.append("elbow_overflexed")

        wrist_ok = True
        if self.stage == "down":
            wrist_ok = avg_wrist < self.WRIST_THRESH
            if not wrist_ok:
                fs -= 10
                self.feedback.append(("Wrist not vertical! Keep forearm straight.", (0,200,255)))
                self._rep_errs.append("wrist_angle")

        arm_ok = True
        if self.stage == "down":
            if avg_arm < self.ARM_POS_MIN:
                arm_ok = False
                fs -= 15
                self.feedback.append(("Raise elbows to shoulder level!", (0,165,255)))
                self._rep_errs.append("elbow_low")
            elif avg_arm > self.ARM_POS_MAX:
                arm_ok = False
                fs -= 15
                self.feedback.append(("Elbows too wide / too high!", (0,165,255)))
                self._rep_errs.append("elbow_flare")

        sym_ok = ediff < self.SYM_THRESH
        if not sym_ok:
            fs -= 15
            side = "Left" if lea < rea else "Right"
            self.feedback.append((f"{side} arm lagging! Press evenly.", (255,165,0)))
            self._rep_errs.append("asymmetry")

        self._rep_sum += max(0, fs)
        self._rep_n += 1

        if avg_elbow < self.ELBOW_DOWN:
            self.stage = "down"

        if avg_elbow > self.ELBOW_UP and self.stage == "down":
            rom = self._max_elbow - self._min_elbow
            if rom < 45:
                self.feedback.append(("Incomplete ROM! Press all the way up.", (0,165,255)))
                self._rep_errs.append("partial_rom")
                fs -= 15
            self.stage = "up"
            score, errs, good = self._finish_rep()
            self._last_rep = {"good": good, "score": score, "errors": errs}
            self._rep_show = 90
            self._min_elbow = 180
            self._max_elbow = 0
            self._depth_ok = False

        if self._rep_show > 0:
            self._rep_show -= 1

        if not self.feedback:
            self.feedback.append(("Perfect form!", (0,255,0)))

        fs = max(0, min(100, fs))
        self.form_scores.append(fs)
        return {
            "left_elbow": lea,    "right_elbow": rea,    "avg_elbow": avg_elbow,
            "left_arm":   laa,    "right_arm":   raa,    "avg_arm":   avg_arm,
            "left_wrist": lw_ang, "right_wrist": rw_ang, "avg_wrist": avg_wrist,
            "ediff": ediff,
            "elbow_ok": elbow_ok, "wrist_ok": wrist_ok,
            "arm_ok": arm_ok,     "sym_ok": sym_ok,
            "form_score": fs, "counter": self.counter, "good_reps": self.good_reps,
            "stage": self.stage,  "feedback": self.feedback,
        }

    def draw_info_panel(self, frame, r):
        self._draw_header(frame, r)
        h, w = frame.shape[:2]

        rows = [
            (f"L Elbow: {r['left_elbow']:.0f}°   R: {r['right_elbow']:.0f}°", 105),
            (f"Avg Elbow: {r['avg_elbow']:.0f}°",                              128),
            (f"L Wrist:  {r['left_wrist']:.1f}°   R: {r['right_wrist']:.1f}°", 155),
            (f"Avg Wrist: {r['avg_wrist']:.1f}°",                              178),
            (f"L ArmPos: {r['left_arm']:.0f}°   R: {r['right_arm']:.0f}°",    205),
            (f"Sym diff: {r['ediff']:.1f}°",                                    228),
        ]
        for txt, y in rows:
            cv2.putText(frame, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200,200,200), 1)

        indicators = [
            ("Elbow Depth",  r["elbow_ok"], 258),
            ("Wrist",        r["wrist_ok"], 283),
            ("Arm Position", r["arm_ok"],   308),
            ("Symmetry",     r["sym_ok"],   333),
        ]
        for lbl, ok, y in indicators:
            c = (0,255,0) if ok else (0,0,255)
            cv2.circle(frame, (20, y-5), 7, c, -1)
            cv2.putText(frame, f"{'OK' if ok else '!!'} {lbl}", (35, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)

        if self._last_rep and self._rep_show > 0:
            res = self._last_rep
            bclr = (0,140,0) if res["good"] else (0,0,180)
            label = (f"REP #{self.counter}: COUNTED ({res['score']}%)"
                     if res["good"] else f"REP #{self.counter}: NOT COUNTED ({res['score']}%)")
            cv2.rectangle(frame, (w//2-220, h-98), (w//2+220, h-65), bclr, -1)
            cv2.putText(frame, label, (w//2-210, h-76),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
            if not res["good"] and res["errors"]:
                names = {
                    "elbow_low": "Elbow Low",
                    "elbow_flare": "Elbow Flare",
                    "elbow_overflexed": "Too Deep",
                    "wrist_angle": "Wrist",
                    "asymmetry": "Asymmetry",
                    "partial_rom": "Partial ROM"
                }
                cv2.putText(frame, "  |  ".join(names.get(e, e) for e in res["errors"]),
                            (w//2-210, h-68), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,200,100), 1)

    def draw_angles(self, frame, landmarks):
        lml = landmarks.landmark
        h, w = frame.shape[:2]
        for key, idx, clr in [
            ("le", PL.LEFT_ELBOW,     (0,255,255)),
            ("re", PL.RIGHT_ELBOW,    (0,255,255)),
            ("la", PL.LEFT_SHOULDER,  (255,200,0)),
            ("ra", PL.RIGHT_SHOULDER, (255,200,0)),
        ]:
            p = lml[idx.value]
            cv2.putText(frame, f"{int(self._last_ang.get(key, 0))}",
                        (int(p.x*w)-15, int(p.y*h)-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, clr, 2)


EXERCISES = {
    "1": ("Barbell Squat",  SquatAnalyzer),
    "2": ("Shoulder Press", ShoulderPressAnalyzer),
}

def print_summary(summary, name):
    if not summary:
        print("\nNo reps performed.")
        return
    print(f"\n{'='*50}\n  {name} — SESSION SUMMARY\n{'='*50}")
    print(f"  Total: {summary['total']}  |  Good: {summary['good']}  |  Avg: {summary['avg_score']:.1f}/100")
    print(f"  {'─'*46}")
    for r in summary["reps"]:
        tag = "GOOD" if r["score"] >= 60 else "LOW"
        errs = ", ".join(r["errors"]) if r["errors"] else "—"
        print(f"  Rep #{r['rep']:2d}  {r['score']:3d}/100  [{tag}]  {errs}")
    print("="*50)

def run(analyzer, cam=0):
    cap = cv2.VideoCapture(cam)
    if not cap.isOpened():
        print("[ERROR] Camera not found!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    name = analyzer.EXERCISE_NAME
    print(f"\n  {name}  |  Q=Quit  R=Reset  P=Pause  S=Summary\n")

    paused = False
    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5, model_complexity=1) as pose:
        while cap.isOpened():
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                res = pose.process(rgb)
                rgb.flags.writeable = True

                if res.pose_landmarks:
                    mp_draw.draw_landmarks(
                        frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                        mp_draw.DrawingSpec(color=(245,66,230),  thickness=2, circle_radius=1)
                    )
                    r = analyzer.analyze_frame(res.pose_landmarks)
                    analyzer.draw_angles(frame, res.pose_landmarks)
                    analyzer.draw_info_panel(frame, r)
                else:
                    cv2.putText(frame, "Pose not detected — Stand in front of camera",
                                (40, frame.shape[0]//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

                cv2.putText(frame, "LIVE", (frame.shape[1]-65, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            else:
                cv2.putText(frame, "PAUSED", (frame.shape[1]//2-80, frame.shape[0]//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 3)

            cv2.imshow(name, frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'), 27):
                break
            elif k == ord('r'):
                analyzer.__init__()
                print("[INFO] Reset.")
            elif k == ord('p'):
                paused = not paused
            elif k == ord('s'):
                print_summary(analyzer.get_summary(), name)

    cap.release()
    cv2.destroyAllWindows()
    print_summary(analyzer.get_summary(), name)

def main():
    print("\n  EXERCISE FORM ANALYSIS\n  ─────────────────────")
    print("  [1] Barbell Squat\n  [2] Shoulder Press\n  [Q] Quit")
    while True:
        c = input("  Choice: ").strip().lower()
        if c == 'q':
            return
        if c in EXERCISES:
            name, cls = EXERCISES[c]
            run(cls())
            return
        print("  Enter 1 or 2.")

if __name__ == "__main__":
    main()
