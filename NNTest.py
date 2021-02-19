import numpy as np
import math
import matplotlib.pyplot as plt
import cv2
import glob
import os



class NNExperiment:
    
    def softmax(self,dot_prod):
        shiftx = dot_prod - np.max(dot_prod)
        exps = np.exp(shiftx)
        return exps / np.sum(exps)

    def softmax_grad(self,dot_prod):
        return dot_prod * ( 1 - dot_prod)
#        Sz = self.softmax(dot_prod)
#        D = -np.outer(Sz, Sz) + np.diag(Sz.flatten())
#        return D

    def ReLU(self,dot_prod):
        return dot_prod * (dot_prod > 0)
    
    def ReLU_derivative(self,dot_prod):
        return 1 * (dot_prod > 0)
        
    
    def sigmoid(self,dot_prod):
        z = 1/(1 + np.exp(-dot_prod))
        return z
           
           
    def sigmoid_derivative(self,layer):
        f = 1/(1 + np.exp(-layer))
        df = f * (1 - f)
        return df


    def __init__(self):
    
        #DECIDING NUMBER OF NODES
        self.num_pixels = 3
        self.layer2_neurons = 2
        self.layer3_neurons = 2
        self.output_neurons = 2
        
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
        self.bias2 = np.zeros(self.layer2_neurons)
        
        #SETTING UP NODES FOR 2ND LAYER
        self.nodes2 = np.zeros(self.layer2_neurons)
        
        
        #SETTING UP A 20 NODES (COLS) AND 200 WEIGHTS (ROWS) MATRIX FOR THE 3RD LAYER
        self.weights3 = np.random.rand(self.layer2_neurons,self.layer3_neurons)
        
        #XAVIER INITIALIZATION
        self.weights3 = self.weights3 * (math.sqrt(1/self.layer2_neurons))
        
        #BIAS CAN ALSO MAKE THIS INTO A MATRIX LATER
        self.bias3 = np.zeros(self.layer3_neurons)
        
        #SETTING UP NODES FOR 3RD LAYER
        self.nodes3 = np.zeros(self.layer3_neurons)
        
        
        
        #SETTING UP A 10 NODES (COLS) AND 20 WEIGHTS (ROWS) MATRIX FOR OUTPUT
        self.weights_output = np.random.rand(self.layer3_neurons,self.output_neurons)
        
        #XAVIER INITIALIZATION
        self.weights_output = self.weights_output * (math.sqrt(1/self.layer3_neurons))
        
        #BIAS CAN ALSO MAKE THIS INTO A MATRIX LATER
        self.bias_output = np.zeros(self.output_neurons)
        
        #SETTING UP NODES FOR OUTPUT LAYER
        self.nodes_output = np.zeros(self.output_neurons)
        
        #WEIGHTS AND BIAS CHANGE
#        self.changes = np.zeros(self.weights_bias_all.shape)

    
    def print_all_weights(self):
        print("INPUT -> 2ND LAYER WEIGHTS")
        print(self.weights2)
        
        print("\n2ND -> 3RD LAYER WEIGHTS")
        print(self.weights3)
        
        print("\n3RD -> OUTPUT LAYER WEIGHTS")
        print(self.weights_output)
    
    
    def print_all_neurons(self):
        print("INPUT -> 2ND LAYER NEURONS")
        print(self.nodes2)
        
        print("\n2ND -> 3RD LAYER NEURONS")
        print(self.nodes3)
        
        print("\n3RD -> OUTPUT LAYER NEURONS")
        print(self.nodes_output)
        
    def print_current_wchange(self):
        print("INPUT -> 2ND LAYER WCHANGE")
        print(self.wchange2)
        
        print("\n2ND -> 3RD LAYER WCHANGE")
        print(self.wchange3)
        
        print("\n3RD -> OUTPUT LAYER WCHANGE")
        print(self.wchange_out)
    
    def print_z(self):
        print("INPUT -> 2ND LAYER NEURONS W/O ACTIVATION")
        print(self.bias2_calc)
        
        print("\n2ND -> 3RD LAYER NEURONS W/O ACTIVATION")
        print(self.bias3_calc)
        
        print("\n3RD -> OUTPUT LAYER NEURONS W/O ACTIVATION")
        print(self.bias_out)
    
    
    def feedforward(self,pixels):
    
        self.pixels_dec = pixels / 1
    
        #2ND LAYER CALCULATION
        #GETTING DOT PRODUCT OF WEIGHTS FOR 2ND LAYER
        self.dotprod2 = np.dot(self.pixels_dec,self.weights2)
    
        #ADDING BIAS TO DOT PRODUCT OF 2ND LAYER
        self.bias2_calc = self.dotprod2 + self.bias2
        
        #SIGMOID CALCULATION FOR NODES OF 2ND LAYER
        self.nodes2 = self.sigmoid(self.bias2_calc)

        

      
        #3RD LAYER CALCULATION
        #GETTING DOT PRODUCT OF WEIGHTS FOR 3RD LAYER
        self.dotprod3 = np.dot(self.nodes2,self.weights3)

        #ADDING BIAS TO DOT PRODUCT OF 3RD LAYER
        self.bias3_calc = self.dotprod3 + self.bias3
        
        #SIGMOID CALCULATION FOR NODES OF 3RD LAYER
        self.nodes3 = self.sigmoid(self.bias3_calc)




        #OUTPUT LAYER CALCULATION
        #GETTING DOT PRODUCT OF WEIGHTS FOR OUTPUT LAYER
        self.dotprod_output = np.dot(self.nodes3,self.weights_output)

        #ADDING BIAS TO DOT PRODUCT OF OUTPUT LAYER
        self.bias_out = self.dotprod_output + self.bias_output

        #SOFT MAX CALCULATION FOR NODES OF OUTPUT LAYER
        self.nodes_output = self.softmax(self.bias_out)
        

        return self.nodes_output
    
        
        
    
    def back_propogation(self,cost_vec,learning_rate):

        
        #LAYER 3-4

        #ERROR CALCULATION (a(L) - y(L))^2
        error = 2 * (cost_vec - self.nodes_output)

        #SOFTMAX DERIVATION
        deriv_softmax = self.softmax_grad(self.nodes_output)

        #WEIGHT CHANGE INITIALIZATION
        self.wchange_out = np.zeros((self.layer3_neurons,self.output_neurons))

        #ALTERNATE WEIGHT CHANGE
        val = (error * deriv_softmax)
        self.wchange_out = np.dot(self.nodes3.reshape(self.layer3_neurons,1),val.reshape(1,self.output_neurons))
        

        #LAYER 2-3

        #SIGMOID DERIVATION
        deriv_sigmoid = self.sigmoid_derivative(self.nodes3)

        #ERROR CALCULATION OF EACH WEIGHT
        error2 = np.zeros((self.layer3_neurons,self.output_neurons))


        #ALTERNATE ERROR CALCULATION 2
        error2 = val * self.weights_output


        #SUM OF ALL THE WEIGHTS ERROR FOR EACH OUTPUT NODE
        error2 = np.sum(error2,axis=1)

        #WEIGHT CHANGE INITIALIZATION
        self.wchange3 = np.zeros((self.layer2_neurons,self.layer3_neurons))


        #ALTERNATE WEIGHT CHANGE
        val2 = (error2 * deriv_sigmoid)
        self.wchange3 = np.dot(self.nodes2.reshape(self.layer2_neurons,1),val2.reshape(1,self.layer3_neurons))
        


        #LAYER 1-2

        #SIGMOID DERIVATION 2
        deriv_sigmoid2 = self.sigmoid_derivative(self.nodes2)

        #ERROR CALCULATION OF EACH WEIGHT
        error3 = np.zeros((self.layer2_neurons,self.layer3_neurons))


        #ALTERNATE ERROR 3 CALCULATION
        error3 = val2 * self.weights3


        #SUM OF ALL WEIGHTS ERROR FOR EACH OUTPUT NODE
        error3 = np.sum(error3,axis=1)

        #WEIGHT CHANGE INITIALIZATION
        self.wchange2 = np.zeros((self.num_pixels,self.layer2_neurons))


        #ALTERNATE WEIGHT CHANGE
        val3 = (error3 * deriv_sigmoid2)
        self.wchange2 = np.dot(self.pixels_dec.reshape(self.num_pixels,1),val3.reshape(1,self.layer2_neurons))



        self.weights_output = self.weights_output - (learning_rate * self.wchange_out)
        self.weights3 = self.weights3 - (learning_rate * self.wchange3)
        self.weights2 = self.weights2 - (learning_rate * self.wchange2)

        #BIAS CALCULATION
#        self.bias_output = self.bias_output - val1
#        self.bias3 = self.bias3 - val2
#        self.bias2 = self.bias2 - val3

        #ALL WEIGHTS AND BIAS FOR EACH LAYER
#        layer_one = np.hstack((self.weights2.flatten(),self.bias2))
#        layer_two = np.hstack((self.weights3.flatten(),self.bias3))
#        layer_three = np.hstack((self.weights_output.flatten(),self.bias_output))
#
#
#        layer_one_two = np.hstack((layer_one,layer_two))
#        all_layers = np.hstack((layer_one_two,layer_three))
#        self.changes = all_layers
#        print(np.sum(self.changes))
#        return self.changes


def main():

    NeuralNet = NNExperiment()

    #PRINTING ORIGINAL WEIGHTS
    print("ORIGINAL XAMIER INITIALIZED WEIGHTS\n")
    NeuralNet.print_all_weights()
    
    #INITIALIZATION OF INPUT ARRAY
    arr = np.array([1,1,0])
    
    #FEEDFORWARD
    NeuralNet.feedforward(arr)
    
    #PRINTING NEURONS WITHOUT ACTIVATION FUNCTION VALUES
    print("\nNEURONS HIDDEN VALUES AFTER FEEDFORWARD BEFORE ACTIVATION\n")
    NeuralNet.print_z()
    
    #PRINTING NEURON HIDDEN VALUES
    print("\nNEURONS HIDDEN VALUES AFTER FEEDFORWARD\n")
    NeuralNet.print_all_neurons()
    
    #BACKPROPAGATION
    NeuralNet.back_propogation(np.array([1,0]),0.5)
    
    #PRINTING WEIGHT CHANGE
    print("\nWEIGHT CHANGE AFTER BACKPROPOGATION\n")
    NeuralNet.print_current_wchange()
    
    #NEW WEIGHTS
    print("\nNEW WEIGHTS\n")
    NeuralNet.print_all_weights()
    
    
    

if __name__ == "__main__":
#    main()
    Sz = np.array([1,2,3])
    print(-np.outer(Sz,Sz))
    print(np.diag(Sz.flatten()))
    D = -np.outer(Sz, Sz) + np.diag(Sz.flatten())
    print(D)
    

    
#    arr = np.array([1,1,0])
#    arr1 = np.array([[1,2],[1,2],[1,2]])
#    print(np.dot(arr,arr1))
