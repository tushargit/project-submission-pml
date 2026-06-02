import cv2
import numpy as np

from src.config import (
    GRID_ROWS,
    GRID_COLS,
    CELL_WIDTH,
    CELL_HEIGHT,
)


class GridSplitter:

    def __init__(self):
        self.rows = GRID_ROWS
        self.cols = GRID_COLS

    def split_into_cells(self, image):
        cells = []

        for row in range(self.rows):
            for col in range(self.cols):

                x1 = col * CELL_WIDTH
                y1 = row * CELL_HEIGHT

                x2 = x1 + CELL_WIDTH
                y2 = y1 + CELL_HEIGHT

                cell = image[y1:y2, x1:x2]
                cells.append(cell)

        return cells

    def get_cell(self, image, cell_index: int):
        cells = self.split_into_cells(image)
        return cells[cell_index]



def draw_grid(image):
    img = image.copy()

    for row in range(1, GRID_ROWS):
        y = row * CELL_HEIGHT
        cv2.line(img, (0, y), (800, y), (0, 255, 0), 1)

    for col in range(1, GRID_COLS):
        x = col * CELL_WIDTH
        cv2.line(img, (x, 0), (x, 600), (0, 255, 0), 1)

    return img