FINGER_TIPS = [8, 12, 16, 20]
FINGER_PIPS = [6, 10, 14, 18]

def classify_gesture(hand_landmarks):
    lm = hand_landmarks

    fingers = []
    fingers.append(lm[4].x < lm[3].x)

    for tip, pip in zip(FINGER_TIPS, FINGER_PIPS):
        fingers.append(lm[tip].y < lm[pip].y)

    if fingers == [False, False, False, False, False]:
        return "fist"
    if fingers == [True, False, False, False, False]:
        return "thumbs_up"
    if fingers == [False, True, True, False, False]:
        return "peace"
    if fingers == [True, True, True, True, True]:
        return "open_hand"
   

    return "unknown"