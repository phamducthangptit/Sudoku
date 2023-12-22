from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import numpy as np
import imutils
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from Sudoku import Sudoku
def find_puzzle(image):


    #Chuyen anh thanh mau xam va su dung Gaussian giam nhieu
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 3)

    #ap dung adaptive thresh
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)

    #Tim duong vien
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    #sap xep theo thu tu giam dan
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    #Khoi tao duong vien
    puzzleCnt = []


    for c in cnts:
        #Tinh sap xi duong vien
        area = cv2.contourArea(c)
        if area < 25000:
            break
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if (len(approx) == 4):
            puzzleCnt = approx
            break

        #neu tim thay thi lay duong vien


    if len(puzzleCnt) == 0:
        return (None, None, puzzleCnt)

    #xu li de lay goc nhin tu tren xuong
    puzzle = four_point_transform(image, puzzleCnt.reshape(4, 2))
    warped = four_point_transform(gray, puzzleCnt.reshape(4, 2))


    return puzzle, warped, puzzleCnt
def extract_digit(cell):
    thresh = cv2.adaptiveThreshold(cell, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 3)
    thresh = clear_border(thresh)

    #Tim duong vien cua cell
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if len(cnts) == 0:
        return None
    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)

    (h, w) = thresh.shape
    percentFilled = cv2.countNonZero(mask) / float(h * w)

    #neu percentFilled < 0.03
    if percentFilled < 0.03:
        return None

    digit = cv2.bitwise_and(thresh, thresh, mask=mask)
    # kernel = np.ones((1, 1), np.uint8)
    # digit = cv2.erode(digit, kernel, iterations=1)

    # cv2.imshow("digit", digit)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return digit

def change_brightness(img, alpha, beta):
    img_new = np.asarray(alpha*img + beta, dtype=int)   # cast pixel values to int
    img_new[img_new>255] = 255
    img_new[img_new<0] = 0
    return img_new


def process_img(warped):
    model = tf.keras.models.load_model('model_number_42.h5')

    board = np.zeros((9, 9), dtype='int')

    print(warped.shape)
    warped = cv2.resize(warped, (450, 450))
    stepX = warped.shape[1] // 9
    stepY = warped.shape[0] // 9
    list_tmp = []
    for y in range(0, 9):
        row = []
        for x in range(0, 9):
            startX = x * stepX
            startY = y * stepY
            endX = (x + 1) * stepX
            endY = (y + 1) * stepY

            row.append((startX, startY, endX, endY))
            cell = warped[startY:endY, startX:endX]

            digit = extract_digit(cell) #lay ra chu so o cell do

            if digit is not None:
                board[y][x] = 1
                x1 = 0
                for i in range(digit.shape[0]):
                    for j in range(digit.shape[1]):
                        if digit[i][j] != 0 :
                            x1 = i
                            #print(f'{x1} lan {i}')
                            break
                    if x1 != 0 :
                        break
                tmp1 = digit.shape[0]
                h = tmp1
                while tmp1 > 0:
                    for j in range(digit.shape[1]):
                        if digit[tmp1-1][j] != 0 :
                            h = tmp1
                            #print(f'{h} lan {tmp1}')
                            break
                    if h != digit.shape[0]:
                        break
                    tmp1 = tmp1 -1
                y1 = 0
                for j in range(digit.shape[1]):
                    for i in range(digit.shape[0]):
                        if digit[i][j] != 0:
                            y1 = j
                            #print(f'{y1} lan {j}')
                            break
                    if y1 != 0:
                        break
                tmp2 = digit.shape[1]
                w = tmp2
                while tmp2 > 0:
                    for j in range(digit.shape[0]):
                        if digit[j][tmp2-1] != 0 :
                            w = tmp2
                            #print(f'{w} lan {tmp2}')
                            break
                    if w != digit.shape[1]:
                        break
                    tmp2 = tmp2 -1
                a = 7
                b = 10
                c = 12
                d = 10
                # a = 5
                # b = 3
                # c = 7
                # d = 6
                x1 = x1 - a if x1-a > 0 else 0
                y1 = y1-c if y1-c>0 else 0
                h = h + b if h +b < digit.shape[0] else digit.shape[0]
                w = w + d if w + d < digit.shape[1] else digit.shape[1]
                digit = digit[x1:h,y1:w]
                digit = cv2.resize(digit,(28,28))
                #digit = change_brightness(digit,5.0,0)

                digit = digit.astype('float32') / 255.0
                digit1 =np.expand_dims(np.array([digit]), -1)
                list_tmp.append(digit1)

                # plt.subplot(9,9,y*9+x+1), plt.imshow(digit), plt.title(pred), plt.axis("off"),plt.subplots_adjust(hspace=1)
    count = len(list_tmp)
    if count  != 0:
        all_preds = model.predict(tf.reshape(np.array(list_tmp), (count, 28, 28, 1)))
        preds = list(map(np.argmax, all_preds))
        a =0
        for i in range(9):
            for j in range(9):
                if board[i][j] == 1:
                    board[i][j] = preds[a] + 1
                    a +=1
    return board

def draw_corners(img, corners):
    for corner in corners:
        x, y = corner
        cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1)
    return img


def dots(time_out_corners):
    multiplier = int(time_out_corners // 1)
    nasobek = int(multiplier/5) +1
    if multiplier > (5 * nasobek):
        nasobek += 1
    tecky = 5 + multiplier - (5 * nasobek)
    return '.' * tecky

def seraching_rectange(img, counter):
    corner_1 = (75 + (2 * counter), 75 + (2 * counter))
    corner_2 = (725 - (2 * counter), 525 - (2 * counter))
    cv2.rectangle(img, corner_1, corner_2, (0, 0, 255), 2)
    return img, corner_1[0]

def get_vars(out_corners_check, solved, bad_read,time_on_corners,seen,solved_text):

    if out_corners_check:
        return "Searching for grid" , (300, 25), (255, 255, 255)
    if not solved and bad_read and time_on_corners > 1:
        return 'model misread digits', (300, 30) ,(0, 0, 255)

    if not (solved or bad_read):
        return "sudoku grid detected", (300, 30), (0, 255, 0)

    if seen and solved and not bad_read and not out_corners_check:
        return solved_text, (285, 30),(0, 255, 0)

    return '', (320, 30) ,(0, 0, 255)

def text_on_top(img, text1, color1, pos1, fps):
    cv2.rectangle(img, (0, 0), (1000, 40), (0, 0, 0), -1)
    cv2.putText(img=img, text=text1, org=pos1, fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1,
                color=color1, thickness=1)
    cv2.putText(img=img, text=f'fps: {fps}', org=(35, 60),
                fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1,
                color=(255,255,255), thickness=1)

    return img
def displayNumbers(img, numbers, solved_num, color=(0, 255, 0)):
    w = int(img.shape[1] / 9)
    h = int(img.shape[0] / 9)

    for i in range(9):
        for j in range(9):
            if numbers[j][i] == 0:
                cv2.putText(img, str(solved_num[j][i]),
                            (i * w + int(w / 2) - int((w / 4)), int((j + 0.7) * h)),
                            cv2.FONT_HERSHEY_COMPLEX, 1, color,
                            1, cv2.LINE_AA)
    return img

def get_inv_perspective(img, masked_num, location, height=450, width=450):

    pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    pts2 = np.float32([location[0], location[1], location[2], location[3]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(masked_num, matrix, (img.shape[1],
                                                      img.shape[0]))
    return result
def inv_transformation(mask,img,predicted_matrix,solved_matrix,corners):
    img_solved = displayNumbers(mask, predicted_matrix, solved_matrix)
    inv = get_inv_perspective(img, img_solved, corners)
    img = cv2.addWeighted(img,0.5, inv,0.5, 0)

    # plt.figure(figsize=(14,10))
    # plt.subplot(131), plt.imshow(img_solved), plt.title("Kết quả"), plt.axis('off')
    # plt.subplot(132), plt.imshow(inv), plt.title("Sau khi phối cảnh"),plt.axis('off')
    # plt.subplot(133), plt.imshow(img), plt.title("Kết quả cuối cùng"),plt.axis('off')
    # plt.show()
    return img,img_solved


def get_corners(contour):

    biggest_contour = contour.reshape(len(contour), 2)
    suma_vekt = biggest_contour.sum(1)
    suma_vekt2 = np.delete(biggest_contour, [np.argmax(suma_vekt), np.argmin(suma_vekt)], 0)

    corners = np.float32([biggest_contour[np.argmin(suma_vekt)], suma_vekt2[np.argmax(suma_vekt2[:, 0])],
                          suma_vekt2[np.argmin(suma_vekt2[:, 0])], biggest_contour[np.argmax(suma_vekt)]])

    return corners

if __name__ == "__main__":
    img = cv2.imread('sudoku2.png')
    # board_sudoku = np.zeros((9, 9), dtype='int')
    (tmp1, warp,tmp2) = find_puzzle(img)
    board_sudoku = process_img(warp)
    # print(board_sudoku)
    check = 0
    for i in range(9):
        for j in range(9):
            if board_sudoku[i][j] != 0:
                check += 1
    if check >= 17:
        board1 = board_sudoku.copy()
        sudoku = Sudoku(board1)
        sudoku.solveSudoku()
        board = np.array(sudoku.board)
        corners = get_corners(tmp2)
        mask = np.zeros((450, 450, 3), dtype=np.uint8)
        img_result, img_solved = inv_transformation(mask, img, board_sudoku, board, corners)
        plt.figure(figsize=(16,14))
        plt.subplot(121), plt.imshow(img), plt.title('Ảnh gốc'), plt.axis('off')
        plt.subplot(122), plt.imshow(img_result), plt.title('Ảnh kết quả'), plt.axis('off')
        plt.show()
    else:
        print("Không thể giải được Sudoku!")

