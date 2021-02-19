import numpy as np
import math
import matplotlib.pyplot as plt
import cv2

def main():

    LEGO_BLOCKS = {"2x2-" : 1, "1x2-" : 1, "1x1-" : 1, "2x2plate-" : 1, "1x2plate-" : 1, "1x1plate-" : 1, "rooftile-" : 1, "peg2-" : 1, "lever-" : 1, "halfbush-" : 1 }
    
    for i in range (0,350):
        for key in LEGO_BLOCKS:
            cv_img = cv2.imread("/Users/arnavsarin/Desktop/NeuralNetwork/25%_ROTATED_TRAINING/" + key +  str(LEGO_BLOCKS.get(key)).zfill(4) + ".png",0)
            print(key + str(i))
            
            #ROTATING IMAGES (ROTATING)
            img_rot_90 = cv2.rotate(cv_img, cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite("/Users/arnavsarin/Desktop/NeuralNetwork/25%_ROTATED_TRAINING/" + key + str(350 + LEGO_BLOCKS.get(key)).zfill(4) + ".png", img_rot_90)
            
            img_rot_180 = cv2.rotate(cv_img, cv2.ROTATE_180)
            cv2.imwrite("/Users/arnavsarin/Desktop/NeuralNetwork/25%_ROTATED_TRAINING/" + key + str(700 + LEGO_BLOCKS.get(key)).zfill(4) + ".png", img_rot_180)
            
            img_rot_270 = cv2.rotate(cv_img, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imwrite("/Users/arnavsarin/Desktop/NeuralNetwork/25%_ROTATED_TRAINING/" + key + str(1050 + LEGO_BLOCKS.get(key)).zfill(4) + ".png", img_rot_270)
            
            
            #FLIPPING IMAGES (MIRRORING)
            img_flip_ud = cv2.flip(cv_img, 0)
            cv2.imwrite("/Users/arnavsarin/Desktop/NeuralNetwork/25%_ROTATED_TRAINING/" + key + str(1400 + LEGO_BLOCKS.get(key)).zfill(4) + ".png", img_flip_ud)
            

            img_flip_lr = cv2.flip(cv_img, 1)
            cv2.imwrite("/Users/arnavsarin/Desktop/NeuralNetwork/25%_ROTATED_TRAINING/" + key + str(1750 + LEGO_BLOCKS.get(key)).zfill(4) + ".png", img_flip_lr)
            

            img_flip_ud_lr = cv2.flip(cv_img, -1)
            cv2.imwrite("/Users/arnavsarin/Desktop/NeuralNetwork/25%_ROTATED_TRAINING/" + key + str(2100 + LEGO_BLOCKS.get(key)).zfill(4) + ".png", img_flip_ud_lr)
            
            
            LEGO_BLOCKS[key] = LEGO_BLOCKS[key] + 1
            
#                pixel_nodes = cv_img.flatten()


if __name__ == "__main__":
    main()
