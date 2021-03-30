import numpy as np
import cv2 as cv
import imageio
import sys
import os
import shutil


def mask_width(mask):
    w = mask.shape[1]

    lower = np.inf
    upper = -np.inf
    for row in mask:
        min_x_val = 0
        max_x_val = w - 1
        while min_x_val < w:
            if np.array_equal(row[min_x_val][:-1], np.array([0, 0])):
                break
            min_x_val += 1

        while max_x_val > 0:
            if np.array_equal(row[max_x_val][:-1], np.array([0, 0])):
                break
            max_x_val -= 1
        if min_x_val < lower:
            lower = min_x_val
        if max_x_val > upper:
            upper = max_x_val
    return upper - lower



if len(sys.argv) != 5:
    # C:\> python seam_cutting.py file.jpg folder 100 -> sys.arg = ["seam_cutting.py", "file.jpg", "folder", "100"]
    print("Benutzung: seam_cutting.py <eingabedatei> <ausgabeordner> <anzahl zu entfernender seams> <maske>")
    sys.exit(0)

# Anzahl an seams, die entfernt werden
C = int(sys.argv[3])

# create the directory
if os.path.exists(sys.argv[2]) and os.path.isdir(sys.argv[2]):
    shutil.rmtree(sys.argv[2])

os.mkdir(sys.argv[2])

# bild geladen
image = cv.imread(sys.argv[1])
mask = cv.imread(sys.argv[4])

mask_w = mask_width(mask)

width, height = image.shape[1], image.shape[0]

weight = np.linalg.norm([255,255,255])* height + 1

for c in range(mask_w):
    print("{}. Durchlauf".format(c + 1))
    costs = np.zeros(shape=[height, width])

    #Array der Pfade mit den geringsten Kosten, Pfad steht immer unter Anfangspixel
    paths = np.empty(shape=[height, width])

    # y koordinate nicht abspeichern, irrelevant

    for y in range(1, height):
        for x in range(width):
            child_pixels = [max(x -1, 0), x , min(width - 1, x + 1)]

            min_cost_pixel = -1
            costs[y, x] = np.inf

            for child in child_pixels:
                new_cost = np.linalg.norm(image[y - 1, child] - image[y, x]) + costs[y - 1, child]
                if np.array_equal(mask[y, x][:-1], np.array([ 0, 0])):
                    new_cost -= weight
                if np.array_equal(mask[y, x][1:], np.array([ 0, 0])):
                    new_cost += weight
                if new_cost < costs[y, x]:
                    costs[y, x] = new_cost
                    min_cost_pixel = child

            paths[y, x] = min_cost_pixel

    min_path_index = np.where(costs[height - 1] == np.amin(costs[height - 1]))[0][0]

    path = [int(min_path_index)]
    for i in range(1,height):
        last = int(path[-1])
        current = int(paths[height - i, last])
        path.append(current)
    #Path sind die x-Werte von unten nach oben

    width = width - 1
    # width -= 1

    new_image = np.empty(shape=[height, width, 3])
    new_mask = np.empty(shape=[height, width, 3])

    y = 0
    for x in reversed(path):
        old_x = 0
        for i in range(width):
            if i == x:
                old_x += 1
            new_image[y, i] = image[y, old_x]
            new_mask[y, i] = mask[y, old_x]
            old_x += 1
        y += 1


    cv.imwrite(sys.argv[2] + "/{}.png".format(c), new_image)

    image = new_image
    mask = new_mask
