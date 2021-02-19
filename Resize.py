import numpy as np
import math
import matplotlib.pyplot as plt
import cv2

def main():

    LEGO_BLOCKS = {"2x2-" : 1, "1x2-" : 1, "1x1-" : 1, "2x2plate-" : 1, "1x2plate-" : 1, "1x1plate-" : 1, "rooftile-" : 1, "peg2-" : 1, "lever-" : 1, "halfbush-" : 1 }
    
    for i in range (0,350):
        for key in LEGO_BLOCKS:
            cv_img = cv2.imread("/Users/arnavsarin/Desktop/NeuralNetwork/350_IMAGES_TO_RESIZE/" + key +  str(LEGO_BLOCKS.get(key)).zfill(4) + ".png",0)
            print(key + str(i))
            scale_percent = 25
            width = int(cv_img.shape[1] * scale_percent / 100)
            height = int(cv_img.shape[0] * scale_percent / 100)
            dsize = (width, height)
            output = cv2.resize(cv_img, dsize)
            cv2.imwrite("/Users/arnavsarin/Desktop/NeuralNetwork/25%_350/" + key + str(LEGO_BLOCKS.get(key)).zfill(4) + ".png", output)
            LEGO_BLOCKS[key] = LEGO_BLOCKS[key] + 1
            
#                pixel_nodes = cv_img.flatten()

if __name__ == "__main__":
    main()
