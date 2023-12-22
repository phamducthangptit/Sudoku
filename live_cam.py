import cv2
import numpy as np
import time as t
from puzzle import *
import tensorflow as tf
import matplotlib.pyplot as plt
from Sudoku import Sudoku




size = (800, 600)
capture = cv2.VideoCapture("https://10.252.2.34:8080/video")
# frame = cv2.imread("123.jpg")
prev = 0
seen = False
steps_mode = False
bad_read = False
solved = False
seen_corners = 0
not_seen_corners = t.time() - 1
wait = 0.4
process_step = 0
rectangle_counter = 0
time_on_corners = 0
dots_str = ""
time = ""
board = []
while capture.isOpened():
    _, frame = capture.read()
    time_elapsed = t.time() - prev
    start = t.time()
    prev = t.time()


    frame = cv2.resize(frame, size)
    img_result = frame.copy()
    # Tìm ra khung sudoku, và toạ độ bốn góc của sudoku
    (ttt, warped, contour) = find_puzzle(frame) # warped = result
    # board_sudoku = np.zeros((9, 9), dtype='int')
    # board_sudoku = process_img(frame)


    if len(contour) == 4:
        # Sắp xếp toạ độ 4 góc theo thứ tự
        corners = get_corners(contour)
        not_seen_corners = 0
        out_corners_check = False
        if not solved:
            if not bad_read:
                color = (0, 0, 255) if int((10 * time_on_corners)) % 3 == 0 else (0, 255, 0)
            cv2.drawContours(img_result, [contour], -1, color, 2)

        else:
            draw_corners(img_result, corners)
        if seen_corners == 0:
            seen_corners = t.time()
        time_on_corners = t.time() - seen_corners
        if time_on_corners > wait:
            wait = 0.4
            dots_str = ''
            if not seen:
                board_sudoku = process_img(warped)
                check = 0
                for i in range(9):
                    for j in range(9):
                        if board_sudoku[i][j] !=0:
                            check +=1
                if check >= 17:
                    board1 = board_sudoku.copy()
                    sudoku = Sudoku(board1)
                    sudoku.solveSudoku()
                    board = np.array(sudoku.board)
                    if np.any(board == 0):
                        bad_read = True
                        solved = False
                    else:
                        bad_read = False
                        seen = True
                        solved = True
                        wait = 0.03
                else:
                    bad_read = True
            if not bad_read:
                # mask = np.zeros_like(perspective_transform(frame, (450, 450), corners))
                mask = np.zeros((450, 450,3), dtype=np.uint8)
                img_result, img_solved = inv_transformation(mask, img_result, board_sudoku, board, corners)

    else:
        if not_seen_corners == 0:
            not_seen_corners = t.time()
        time_out_corners = t.time() - not_seen_corners
        out_corners_check = time_out_corners > 0.2
        if out_corners_check:
            dots_str = dots(time_out_corners)
            seen = False
            seen_corners = 0
            bad_read = False
            solved = False
            wait = 0.4
            board = []
            img_result, corner_rect = seraching_rectange(img_result, rectangle_counter)
            # helping variable for searching rectangle
            if corner_rect > 200:
                rectangle_counter = -1
            rectangle_counter += 1
    # text writing
    fps = int(1 / time_elapsed)
    text, pos, color1 = get_vars(out_corners_check,solved,bad_read,time_on_corners,seen,time)
    text_on_top(img_result, text + dots_str, color1, pos, fps)


    cv2.imshow('sudoku solver', img_result)
    key = cv2.waitKey(1)

    if int(key) == 113:
            break

capture.release()  # giải phóng tài nguyên
cv2.destroyAllWindows()

