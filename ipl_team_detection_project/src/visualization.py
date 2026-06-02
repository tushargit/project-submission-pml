import cv2

from src.config import (
    CELL_WIDTH,
    CELL_HEIGHT,
    GRID_ROWS,
    GRID_COLS
)

CLASS_NAMES = {
    0: "No Team",
    1: "CSK",
    2: "DC",
    3: "GT",
    4: "KKR",
    5: "LSG",
    6: "MI",
    7: "PBKS",
    8: "RR",
    9: "RCB",
    10: "SRH"
}

TEAM_COLORS = {
    0: (120,120,120),
    1: (0,255,255),     # CSK
    2: (255,0,0),       # DC
    3: (0,128,255),     # GT
    4: (128,0,128),     # KKR
    5: (255,255,0),     # LSG
    6: (255,0,255),     # MI
    7: (0,0,255),       # PBKS
    8: (255,128,128),   # RR
    9: (0,255,0),       # RCB
    10:(255,165,0)      # SRH
}

def draw_predictions(image, predictions):

    img = image.copy()

    idx = 0

    for row in range(GRID_ROWS):

        for col in range(GRID_COLS):

            x1 = col * CELL_WIDTH
            y1 = row * CELL_HEIGHT

            x2 = x1 + CELL_WIDTH
            y2 = y1 + CELL_HEIGHT

            label = int(predictions[idx])

            # COLOR
            color = TEAM_COLORS.get(label, (255, 255, 255))
            # TRANSPARENT OVERLAY
            overlay = img.copy()

            cv2.rectangle(
                overlay,
                (x1, y1),
                (x2, y2),
                color,
                -1
            )

            # BLEND OVERLAY
            cv2.addWeighted(
                overlay,
                0.25,
                img,
                0.75,
                0,
                img
            )

            # BORDER
            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            # TEXT
            text = CLASS_NAMES[label]

            # TEXT SIZE
            (text_width, text_height), _ = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                1
            )

            # BLACK BACKGROUND RECTANGLE
            cv2.rectangle(
                img,
                (x1, y1),
                (x1 + text_width + 6, y1 + text_height + 8),
                (0, 0, 0),
                -1
            )

            # WHITE TEXT
            cv2.putText(
                img,
                text,
                (x1 + 3, y1 + text_height + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            idx += 1

    return img
