import numpy as np
import math
import matplotlib.pyplot as plt
import cv2
import glob
import os
import openpyxl


#MULTIPLYING WEIGHTS TO EACH PIXEL IT CORRESPONDS TO
#EACH WEIGHT SHOULD BE BETWEEN 0-1
#GOAL IS THE MORE APPARENT A PIXEL VALUE IS THE CLOSER THE WEIGHT SHOULD BE TO 1
#IF PIXEL IS DARKER THEN WEIGHT SHOULD BE CLOSER TO 0
#SIGMOID FUNCTION ACCOMPLISHES THIS PUT IN (PIXEL VALUES * WEIGHTS) TO GET NEW WEIGHTS THAT ARE CLOSER TO 0 OR 1
#NEURON_SUM = σ(w1a1 + w2a2 + w3a3 + wnan + BIAS)
#w are the weights, a is the pixel value
#BIAS IS HOW POSITIVE THE NEURON_SUM SHOULD BE MINIMUM TO BE CONSIDERED ACTIVE. ONLY ACTIVE IF NEURON_SUM > BIAS


class NNExperiment:
    
    def softmax(self,dot_prod):
#        e = np.exp(dot_prod)
#        return e / e.sum()
        shiftx = dot_prod - np.max(dot_prod)
        exps = np.exp(shiftx)
        return exps / np.sum(exps)

    def softmax_grad(self,dot_prod):
#        return dot_prod * ( 1 - dot_prod)
        Sz = self.softmax(dot_prod)
        D = -np.outer(Sz, Sz) + np.diag(Sz.flatten())
        return D

    def ReLU(self,dot_prod):
        return dot_prod * (dot_prod > 0)
    
    def ReLU_derivative(self,dot_prod):
        return 1 * (dot_prod > 0)
        
    
    def sigmoid(self,dot_prod):
#        z = 1/(1 + np.exp(-dot_prod))
#        return z
        return np.tanh(dot_prod)
           
           
    def sigmoid_derivative(self,layer):
#        f = 1/(1 + np.exp(-layer))
#        df = f * (1 - f)
#        return df
        dt=1-(self.sigmoid(layer)**2)
        return dt


    def __init__(self):
    
        #DECIDING NUMBER OF NODES
        self.num_pixels = 2500
        self.layer2_neurons = 23
        self.layer3_neurons = 23
        self.output_neurons = 10
        
        #EXTRA VALUES FOR INDEXING
        self.total_weights_1_2 = self.num_pixels * self.layer2_neurons
        self.total_weights_2_3 = self.layer2_neurons * self.layer3_neurons
        self.total_weights_3_out = self.layer3_neurons * self.output_neurons
        
        self.tot_weights_bias_1_2 = self.total_weights_1_2 + self.layer2_neurons
        self.tot_weights_bias_2_3 = self.total_weights_2_3 + self.layer3_neurons
        self.tot_weights_bias_3_out = self.total_weights_3_out + self.output_neurons
        
        
        #SETTING UP A 200 NODES (COLS) AND 40,000 WEIGHTS (ROWS) MATRIX FOR THE 2ND LAYER
        self.weights2 = np.random.rand(self.num_pixels,self.layer2_neurons)
        
        #XAVIER INITIALIZATION
        self.weights2 = self.weights2 * (math.sqrt(1/self.num_pixels))
        
        #BIAS CAN ALSO MAKE THIS INTO A MATRIX LATER
#        self.bias2 = np.random.rand(self.layer2_neurons)
#        self.bias2 = np.ones(self.layer2_neurons)
        self.bias2 = np.zeros(self.layer2_neurons)

        
        #SETTING UP NODES FOR 2ND LAYER
        self.nodes2 = np.zeros(self.layer2_neurons)
        
        
        #SETTING UP A 20 NODES (COLS) AND 200 WEIGHTS (ROWS) MATRIX FOR THE 3RD LAYER
        self.weights3 = np.random.rand(self.layer2_neurons,self.layer3_neurons)
        
        #XAVIER INITIALIZATION
        self.weights3 = self.weights3 * (math.sqrt(1/self.layer2_neurons))
        
        #BIAS CAN ALSO MAKE THIS INTO A MATRIX LATER
#        self.bias3 = np.random.rand(self.layer3_neurons)
#        self.bias3 = np.ones(self.layer3_neurons)
        self.bias3 = np.zeros(self.layer3_neurons)
        
        #SETTING UP NODES FOR 3RD LAYER
        self.nodes3 = np.zeros(self.layer3_neurons)
        
        
        
        #SETTING UP A 10 NODES (COLS) AND 20 WEIGHTS (ROWS) MATRIX FOR OUTPUT
        self.weights_output = np.random.rand(self.layer3_neurons,self.output_neurons)
        
        #XAVIER INITIALIZATION
        self.weights_output = self.weights_output * (math.sqrt(1/self.layer3_neurons))
        
        #BIAS CAN ALSO MAKE THIS INTO A MATRIX LATER
#        self.bias_output = np.random.rand(self.output_neurons)
#        self.bias_output = np.ones(self.output_neurons)
        self.bias_output = np.zeros(self.output_neurons)
        
        #SETTING UP NODES FOR OUTPUT LAYER
        self.nodes_output = np.zeros(self.output_neurons)
        
        
        #MATRIX WITH ALL WEIGHTS AND BIASES
        weights_bias2 = np.hstack((self.weights2.flatten(), self.bias2))
        weights_bias3 = np.hstack((self.weights3.flatten(), self.bias3))
        weights_bias_output = np.hstack((self.weights_output.flatten(), self.bias_output))

        weights_bias2_3 = np.hstack((weights_bias2,weights_bias3))

        self.weights_bias_all = np.hstack((weights_bias2_3, weights_bias_output))
        
        #WEIGHTS AND BIAS CHANGE
        self.changes = np.zeros(self.weights_bias_all.shape)

    
    def get_hlayer2_neurons(self):
        return self.layer2_neurons
        
    def get_hlayer3_neurons(self):
        return self.layer3_neurons
    
    def feedforward(self,pixels):
    
        self.pixels_dec = pixels / 255
    
        #2ND LAYER CALCULATION
        #GETTING DOT PRODUCT OF WEIGHTS FOR 2ND LAYER
        self.dotprod2 = np.dot(self.pixels_dec,self.weights2)
    
        #ADDING BIAS TO DOT PRODUCT OF 2ND LAYER
        self.bias2_calc = self.dotprod2 + self.bias2
        
        #SIGMOID CALCULATION FOR NODES OF 2ND LAYER
        self.nodes2 = self.sigmoid(self.bias2_calc)

        #ReLU CALCULATION FOR NODES OF 2ND LAYER
#        self.nodes2 = self.ReLU(self.bias2_calc)


      
        #3RD LAYER CALCULATION
        #GETTING DOT PRODUCT OF WEIGHTS FOR 3RD LAYER
        self.dotprod3 = np.dot(self.nodes2,self.weights3)

        #ADDING BIAS TO DOT PRODUCT OF 3RD LAYER
        self.bias3_calc = self.dotprod3 + self.bias3
        
        #SIGMOID CALCULATION FOR NODES OF 3RD LAYER
        self.nodes3 = self.sigmoid(self.bias3_calc)

        #ReLU CALCULATION FOR NODES OF 3RD LAYER
#        self.nodes3 = self.ReLU(self.bias3_calc)



        #OUTPUT LAYER CALCULATION
        #GETTING DOT PRODUCT OF WEIGHTS FOR OUTPUT LAYER
        self.dotprod_output = np.dot(self.nodes3,self.weights_output)

        #ADDING BIAS TO DOT PRODUCT OF OUTPUT LAYER
        self.bias_out = self.dotprod_output + self.bias_output

        #SIGMOID CALCULATION FOR NODES OF OUTPUT LAYER
#        self.nodes_output = self.sigmoid(self.bias_out)
        
        #ReLU CALCULATION FOR NODES OF 3RD LAYER
#        self.nodes_output = self.ReLU(self.bias_out)

        #SOFT MAX CALCULATION FOR NODES OF OUTPUT LAYER
        self.nodes_output = self.softmax(self.bias_out)
        
#        print(self.nodes_output)

        return self.nodes_output
    
        
    
    def back_propogation(self,cost_vec,learning_rate):

        
        #LAYER 3-4

        #ERROR CALCULATION (a(L) - y(L))^2
#        error = 2 * (cost_vec - self.nodes_output)
        error = -1 * (cost_vec / self.nodes_output)

        #SOFTMAX DERIVATION
        deriv_softmax = self.softmax_grad(self.bias_out)

        #WEIGHT CHANGE INITIALIZATION
        self.wchange_out = np.zeros((self.layer3_neurons,self.output_neurons))

        #ALTERNATE WEIGHT CHANGE
#        print(deriv_softmax.shape)
        val = (error * deriv_softmax)
        val_sum = np.sum(val,axis=1)
        self.wchange_out = np.dot(self.nodes3.reshape(self.layer3_neurons,1),val_sum.reshape(1,self.output_neurons))
        
        #ALTERNATE BIAS CHANGE
        self.bchange_out = val_sum
        
#        for j in range (self.nodes_output.size):
#            for k in range (self.nodes3.size):
#                #ERROR * (SOFTMAX DERIVATION LAYER 3) * (LAYER 3 NEURON)
#                self.wchange_out[k][j] = error[j] * deriv_softmax[j] * self.nodes3[k]

#        print(np.sum(self.wchange_out))
#        print(self.wchange_out)

        #LAYER 2-3

        #SIGMOID DERIVATION
        deriv_sigmoid = self.sigmoid_derivative(self.nodes3)
#        deriv_sigmoid = self.ReLU_derivative(self.nodes3)

        #ERROR CALCULATION OF EACH WEIGHT
        error2 = np.zeros((self.layer3_neurons,self.output_neurons))


        #ALTERNATE ERROR CALCULATION 2
        #error = val * self.weights_output
        error2 = val_sum * self.weights_output

        #ERROR CALCULATION 2 ADDITION OF ALL OUTPUT NODES ERROR
#        for j in range (self.nodes_output.size):
#            for k in range (self.nodes3.size):
#                error2[k][j] = error[j] * deriv_softmax[j] * self.weights_output[k][j]



        #SUM OF ALL THE WEIGHTS ERROR FOR EACH OUTPUT NODE
        error2 = np.sum(error2,axis=1)

        #WEIGHT CHANGE INITIALIZATION
        self.wchange3 = np.zeros((self.layer2_neurons,self.layer3_neurons))


        #ALTERNATE WEIGHT CHANGE
        val2 = (error2 * deriv_sigmoid)
        self.wchange3 = np.dot(self.nodes2.reshape(self.layer2_neurons,1),val2.reshape(1,self.layer3_neurons))
        
        #ALTERNATE BIAS CHANGE
        self.bchange3 = val2
        
#        for j in range (self.nodes3.size):
#            for k in range (self.nodes2.size):
#                #ERROR2 * (SIGMOID DERIVATION LAYER 2) * (LAYER 2 NEURON)
#                self.wchange3[k][j] = error2[j] * deriv_sigmoid[j] * self.nodes2[k]


#        print(np.sum(self.wchange3))
#        print(self.wchange3)


        #LAYER 1-2

        #SIGMOID DERIVATION 2
        deriv_sigmoid2 = self.sigmoid_derivative(self.nodes2)
#        deriv_sigmoid2 = self.ReLU_derivative(self.nodes2)

        #ERROR CALCULATION OF EACH WEIGHT
        error3 = np.zeros((self.layer2_neurons,self.layer3_neurons))


        #ALTERNATE ERROR 3 CALCULATION
        error3 = val2 * self.weights3


        #ERROR CALCULATION 3 ADDITION OF ALL OUTPUT NODES ERROR
#        for j in range (self.nodes3.size):
#            for k in range (self.nodes2.size):
#                error3[k][j] = error2[j] * deriv_sigmoid[j] * self.weights3[k][j]


        #SUM OF ALL WEIGHTS ERROR FOR EACH OUTPUT NODE
        error3 = np.sum(error3,axis=1)

        #WEIGHT CHANGE INITIALIZATION
        self.wchange2 = np.zeros((self.num_pixels,self.layer2_neurons))


        #ALTERNATE WEIGHT CHANGE
        val3 = (error3 * deriv_sigmoid2)
        self.wchange2 = np.dot(self.pixels_dec.reshape(self.num_pixels,1),val3.reshape(1,self.layer2_neurons))
        
        #ALTERNATE BIAS CHANGE
        self.bchange2 = val3

#        for j in range (self.nodes2.size):
#            for k in range (self.pixels_dec.size):
#                #ERROR3 * (SIGMOID DERIVATION LAYER) * (INPUT NEURONS)
#                self.wchange2[k][j] = error3[j] * deriv_sigmoid2[j] * self.pixels_dec[k]


#        print(np.sum(self.wchange2))
#        print(self.wchange2)

        #WEIGHT CALCULATION
        self.weights_output = self.weights_output - (learning_rate * self.wchange_out)
        self.weights3 = self.weights3 - (learning_rate * self.wchange3)
        self.weights2 = self.weights2 - (learning_rate * self.wchange2)

        #BIAS CALCULATION
        self.bias_output = self.bias_output - (learning_rate * self.bchange_out)
        self.bias3 = self.bias3 - (learning_rate * self.bchange3)
        self.bias2 = self.bias2 - (learning_rate * self.bchange2)

        #ALL WEIGHTS AND BIAS FOR EACH LAYER
        layer_one = np.hstack((self.weights2.flatten(),self.bias2))
        layer_two = np.hstack((self.weights3.flatten(),self.bias3))
        layer_three = np.hstack((self.weights_output.flatten(),self.bias_output))


        layer_one_two = np.hstack((layer_one,layer_two))
        all_layers = np.hstack((layer_one_two,layer_three))
        self.changes = all_layers
#        print(np.sum(self.changes))
        return self.changes
    
    
    #USED TO AVERAGE CHANGES
    def average_changes(self):
        self.changes = np.delete(self.changes, (0), axis=0)
        self.changes = np.sum(self.changes, axis = 0)
        self.changes = self.changes / 40
#        return self.changes
    
    
    def add_changes(self):
    
        #ADDED CHANGES TO WEIGHTS
        self.weights_bias_all = self.changes
        
        #UPDATED WEIGHTS AND BIASES
        self.weights2 = self.weights_bias_all[0:(self.total_weights_1_2)].reshape(self.num_pixels,self.layer2_neurons)
        self.bias2 = self.weights_bias_all[self.total_weights_1_2:self.tot_weights_bias_1_2]
        self.weights3 = self.weights_bias_all[self.tot_weights_bias_1_2:(self.tot_weights_bias_1_2+self.total_weights_2_3)].reshape(self.layer2_neurons,self.layer3_neurons)
        self.bias3 = self.weights_bias_all[(self.tot_weights_bias_1_2+self.total_weights_2_3):(self.tot_weights_bias_1_2+self.tot_weights_bias_2_3)]
        self.weights_output = self.weights_bias_all[(self.tot_weights_bias_1_2+self.tot_weights_bias_2_3):(self.tot_weights_bias_1_2+self.tot_weights_bias_2_3+self.total_weights_3_out)].reshape(self.layer3_neurons,self.output_neurons)
        self.bias_output = self.weights_bias_all[(self.tot_weights_bias_1_2+self.tot_weights_bias_2_3+self.total_weights_3_out):((self.tot_weights_bias_1_2+self.tot_weights_bias_2_3+self.tot_weights_bias_3_out))]
        
        return self.weights_bias_all
    
    def cost_func(self,expected,actual):
    
        #SUBTRACTING THE TWO MATRICES
        subtract_matrix = np.subtract(expected,actual)
        
        #SQUARING THE MATRIX AND DIVIDING BY 2
        squared_matrix = np.square(subtract_matrix) / 2
        
        #FINDING SUM OF MATRIX FOR TOTAL ERROR
        sum_values =  np.sum(squared_matrix)
        
        return sum_values
    
    
    def label(self,name):
        #LABELING EXPECTED OUTPUT VECTOR FOR BACKPROPOGATION
        self.final_expected_result = np.zeros(10)
        
        if("halfbush-" in name):
            self.final_expected_result[9] = 1.0
        elif("lever-" in name):
            self.final_expected_result[8] = 1.0
        elif("peg2-" in name):
            self.final_expected_result[7] = 1.0
        elif("rooftile-" in name):
            self.final_expected_result[6] = 1.0
        elif("1x1plate-" in name):
            self.final_expected_result[5] = 1.0
        elif("1x2plate-" in name):
            self.final_expected_result[4] = 1.0
        elif("2x2plate-" in name):
            self.final_expected_result[3] = 1.0
        elif("1x1-" in name):
            self.final_expected_result[2] = 1.0
        elif("1x2-" in name):
            self.final_expected_result[1] = 1.0
        elif("2x2-" in name):
            self.final_expected_result[0] = 1.0
            
        return self.final_expected_result
    
    
    def determine_label(self,output):
        
        #DETERMINE INDEX OF LABEL
        index_of_label = np.argmax(output)
#        print(output)
#        print(index_of_label)
            
        if(index_of_label == 0):
            return "2x2-"
        elif(index_of_label == 1):
            return "1x2-"
        elif(index_of_label == 2):
            return "1x1-"
        elif(index_of_label == 3):
            return "2x2plate-"
        elif(index_of_label == 4):
            return "1x2plate-"
        elif(index_of_label == 5):
            return "1x1plate-"
        elif(index_of_label == 6):
            return "rooftile-"
        elif(index_of_label == 7):
            return "peg2-"
        elif(index_of_label == 8):
            return "lever-"
        elif(index_of_label == 9):
            return "halfbush-"
        
            
    
    
    
    def save_parameters(self,parameters, LEGO_BLOCKS):
    
        #TO SAVE THE PARAMETERS
        np.save('parameters.npy', parameters)
        
        #TO SAVE LEGO_BLOCKS PICTURE BATCH
        np.save('lego_blocks.npy', LEGO_BLOCKS)
        
        
    def load_parameters(self):

        try:
            #LOAD WEIGHTS AND BIAS
            self.weights_bias_all = np.load('parameters.npy',allow_pickle=True)
            
            #LOAD LEGO_BLOCKS BATCH
            LEGO_BLOCKS = np.load('lego_blocks.npy',allow_pickle=True)
        
            #SET WEIGHTS AND BIAS
            self.weights2 = self.weights_bias_all[0:self.total_weights_1_2].reshape(self.num_pixels,self.layer2_neurons)
            self.bias2 = self.weights_bias_all[self.total_weights_1_2:self.tot_weights_bias_1_2]
            self.weights3 = self.weights_bias_all[self.tot_weights_bias_1_2:(self.tot_weights_bias_1_2+self.total_weights_2_3)].reshape(self.layer2_neurons,self.layer3_neurons)
            self.bias3 = self.weights_bias_all[(self.tot_weights_bias_1_2+self.total_weights_2_3):(self.tot_weights_bias_1_2+self.tot_weights_bias_2_3)]
            self.weights_output = self.weights_bias_all[(self.tot_weights_bias_1_2+self.tot_weights_bias_2_3):(self.tot_weights_bias_1_2+self.tot_weights_bias_2_3+self.total_weights_3_out)].reshape(self.layer3_neurons,self.output_neurons)
            self.bias_output = self.weights_bias_all[(self.tot_weights_bias_1_2+self.tot_weights_bias_2_3+self.total_weights_3_out):((self.tot_weights_bias_1_2+self.tot_weights_bias_2_3+self.tot_weights_bias_3_out))]
            
            
            #RETURN BATCH
            return LEGO_BLOCKS.tolist()
            
        except OSError as e:
        
            print("LOADED FILE DOESNT EXIST")
            
            #RETURN BATCH
            return {"2x2-" : 1, "1x2-" : 1, "1x1-" : 1, "2x2plate-" : 1, "1x2plate-" : 1, "1x1plate-" : 1, "rooftile-" : 1, "peg2-" : 1, "lever-" : 1, "halfbush-" : 1 }
        
        
    def delete_parameters(self, file_path):
        
        try:
            #DELETE PARAMETERS FILE
            os.remove('parameters.npy')
            
            #DELETE LEGO_BLOCKS FILE
            os.remove('lego_blocks.npy')
            
            #DELETE EXTRA FILE
            os.rmdir(file_path)
            
        except OSError as e:
            print("FILE DOESNT EXIST FOR DELETION")
    
    
    
def main():

    LEGO_BLOCKS = {"2x2-" : 1, "1x2-" : 1, "1x1-" : 1, "2x2plate-" : 1, "1x2plate-" : 1, "1x1plate-" : 1, "rooftile-" : 1, "peg2-" : 1, "lever-" : 1, "halfbush-" : 1 }

    #STEP 1: CONVERT PIXELS OF IMAGE TO 1ST LAYER OF NODES
    
    NeuralNet = NNExperiment()
    
    end = False
    
    file_path = ""
    
    while(not end):
        LEGO_BLOCKS = NeuralNet.load_parameters()
        
        #GET COMMAND
        inp = input("\nSELECT COMMAND: T (TRAIN), TO (TEST ONE), TA (TEST ALL), D (DELETE), S (STOP), DS (DELETE AND STOP)\n")
        
        #TO TRAIN NEURAL NETWORK
        if (inp.strip().upper()== "T"):
            
            learning_rate = 0.1
            
            #ALL BATCHES
            for i in range (0,2450):
                for key in LEGO_BLOCKS:
                    cv_img = cv2.imread("/Users/arnavsarin/Desktop/NeuralNetwork/25%_ROTATED_TRAINING/" + key +  str(LEGO_BLOCKS.get(key)).zfill(4) + ".png",0)
                    pixel_nodes = cv_img.flatten()
                    LEGO_BLOCKS[key] = LEGO_BLOCKS[key] + 1


                    actual = NeuralNet.feedforward(pixel_nodes)
                    expect = NeuralNet.label(key)
                    cost = NeuralNet.cost_func(expect,actual)
                    new_parameters = NeuralNet.back_propogation(expect,learning_rate)
                    print(key + str(LEGO_BLOCKS[key] - 1))

#                NeuralNet.average_changes()
#                new_parameters = NeuralNet.add_changes()
            
            NeuralNet.save_parameters(new_parameters,LEGO_BLOCKS)
        
        #TO TEST NEURAL NETWORK
        elif (inp.strip().upper()== "TO"):
            print("TESTING NEURAL NETWORK")
            
            #ENTER IMAGE INFORMATION
            image_name = input("\nENTER IMAGE FILE NAME:\n")
            
            cv_img2 = cv2.imread("/Users/arnavsarin/Desktop/NeuralNetwork/25%_50/" + image_name + ".png",0)
            pix_nodes = cv_img2.flatten()
            
            NN_output = NeuralNet.feedforward(pix_nodes)
            print(NN_output)
            result = NeuralNet.determine_label(NN_output)
            print("RESULT " + result)
         
         
        elif (inp.strip().upper() == "TA"):
            
            LEGO_BLOCKS_TEST_ALL = {"2x2-" : 1, "1x2-" : 1, "1x1-" : 1, "2x2plate-" : 1, "1x2plate-" : 1, "1x1plate-" : 1, "rooftile-" : 1, "peg2-" : 1, "lever-" : 1, "halfbush-" : 1 }
                       
            wbk = openpyxl.load_workbook("/Users/arnavsarin/Desktop/NeuralNetwork/PARAMETER_DATA.xlsx")
            sheet = wbk.worksheets[0]
            
            count_correct = 0.0
            
            for i in range (0,350):
                for key in LEGO_BLOCKS_TEST_ALL:
                    cv_img_a = cv2.imread("/Users/arnavsarin/Desktop/NeuralNetwork/25%_350/" + key + str(LEGO_BLOCKS_TEST_ALL.get(key)).zfill(4) + ".png",0)
                    pix_a = cv_img_a.flatten()
                    NN_output_a = NeuralNet.feedforward(pix_a)
                    result_a = NeuralNet.determine_label(NN_output_a)
                    if(result_a == key):
                        count_correct = count_correct + 1.0
                    LEGO_BLOCKS_TEST_ALL[key] = LEGO_BLOCKS_TEST_ALL[key] + 1
            
            mrow = sheet.max_row+1
            
            sheet["A" + str(mrow)] = NeuralNet.get_hlayer2_neurons()
            sheet["B" + str(mrow)] = NeuralNet.get_hlayer3_neurons()
            
            percent1 = round((count_correct/3500),6)*100.0
            print("MODEL IS " + str(format(percent1,'.3f')) + "% ACCURATE FOR OLD DATA" )
            sheet["C" + str(mrow)] = float(format(percent1,'.3f'))
            
            count_correct2 = 0.0
            
            for i in range (0,50):
                for key in LEGO_BLOCKS_TEST_ALL:
                    cv_img_a = cv2.imread("/Users/arnavsarin/Desktop/NeuralNetwork/25%_50/" + key + str(LEGO_BLOCKS_TEST_ALL.get(key)).zfill(4) + ".png",0)
                    pix_a = cv_img_a.flatten()
                    NN_output_a = NeuralNet.feedforward(pix_a)
                    result_a = NeuralNet.determine_label(NN_output_a)
                    if(result_a == key):
                        count_correct2 = count_correct2 + 1.0
                    LEGO_BLOCKS_TEST_ALL[key] = LEGO_BLOCKS_TEST_ALL[key] + 1
              
              
            percent2 = round((count_correct2/500),6)*100.0
            print("MODEL IS " + str(format(percent2,'.3f')) + "% ACCURATE FOR NEW DATA" )
            sheet["D" + str(mrow)] = float(format(percent2,'.3f'))
            
            count_correct_all = count_correct + count_correct2
            percent3 = round((count_correct_all/4000.0),6)*100.0
            print("MODEL IS " + str(format(percent3,'.3f')) + "% ACCURATE FOR ALL DATA" )
            sheet["E" + str(mrow)] = float(format(percent3,'.3f'))

            
            wbk.save("PARAMETER_DATA.xlsx")
            wbk.close()
            
            file_path = "/Users/arnavsarin/Desktop/NeuralNetwork/" + str(format(percent3,'.3f')) + "%_" + str(NeuralNet.get_hlayer2_neurons()) + "_" +str(NeuralNet.get_hlayer3_neurons())
            try:
                os.mkdir(file_path)
            except OSError as e:
                break
            
            
        elif (inp.strip().upper()== "D"):
            print("DELETING NPY FILES")
            print(file_path)
            NeuralNet.delete_parameters(file_path)
            
        elif (inp.strip().upper()== "S"):
            end = True
    
        elif (inp.strip().upper()== "DS"):
            print("DELETING NPY FILES")
            print(file_path)
            NeuralNet.delete_parameters(file_path)
            end = True
#            print(np.sum(NeuralNet.feedforward(pixel_nodes)))
            
            
    
    

if __name__ == "__main__":
    main()
#    arnie = np.array([[1,1,1],[1,1,1]])
#    the = np.array([1,2,3])
#    print(arnie * the)
    
    
    

        


