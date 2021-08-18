import cv2
import numpy as np
import operator
from keras.models import load_model, model_from_json
import sudoku_solver as sol

classifier = load_model("./digit_model.h5")

margin = 4
box = 28 + 2 *margin
grid_size = 9 * box

vid = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
flag = 0
out = cv2.VideoWriter('output.avi', fourcc, 30.0, (1080, 620))

while True:

    ret, frame = vid.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(7,7),0)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 2)

    contours, heirarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_grid = None
    maxArea = 0 

    for c in contours:
        area = cv2.contourArea(c)
        if area > 25000:
            peri = cv2.arcLength(c, True)
            polygon = cv2.approxPolyDP(c, 0.01 * peri, True)
            if area > maxArea and len(polygon) == 4:
                contour_grid = polygon
                maxArea = area    

        if contour_grid is not None:
            cv2.drawContours(frame,[contour_grid],0,(0,255,0),2)
            points = np.vstack(contour_grid).squeeze()
            points=sorted(points,key=operator.itemgetter(1))
            if points[0][0]<points[1][0]:
                if points[3][0] < points[2][0]:
                    pts1 = np.float32([points[0], points[1], points[3], points[2]])
                else:
                    pts1 = np.float32([points[0], points[1], points[2], points[3]])
            else:
                if points[3][0] < points[2][0]:
                    pts1 = np.float32([points[1], points[0], points[3], points[2]])
                else:
                    pts1 = np.float32([points[1], points[0], points[2], points[3]])            
            pts2 = np.float32([[0, 0], [grid_size, 0], [0, grid_size], [grid_size, grid_size]])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            grid= cv2.warpPerspective(frame,M,(grid_size,grid_size))
            grid= cv2.cvtColor(grid, cv2.COLOR_BGR2GRAY)
            grid = cv2.adaptiveThreshold(grid, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 3)

            cv2.imshow("grid", grid)
            if flag==0:
                grid_text = []
                for y in range(9):
                    line = ""
                    for x in range(9):
                        y2_min = y* box + margin
                        y2_max= (y+1) * box - margin
                        x2_min = x * box+ margin 
                        x2_max = (x+1)* box-margin
                        cv2.imwrite("mat"+str(y)+str(x)+".png", grid[y2_min:y2_max, x2_min:x2_max])
                        img = grid[y2_min:y2_max, x2_min:x2_max]
                        x=img.reshape(1,28,28,1)
                        if x.sum()>10000:
                            prediction = classifier.predict_classes(x)
                            line += "{:d}".format(prediction[0])
                        else:
                            line += "{:d}".format(0)
                    grid_text.append(line)
                print(grid_text)
                result= sol.sudoku(grid_text)
            print("Resultant:",result)