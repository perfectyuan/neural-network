#创建一个神经网络类
class neuralNetwork :
    
    #initialise the neural network
    def __init__(self,inputnodes,hiddiennotes,outputnodes,learningrate):
        #set number of nodes in each input,hidden,output layer
        self.inodes = inputnodes
        self.hnodes = hiddiennotes
        self.onodes = outputnodes

        #learning rate
        self.lr = learningrate

        pass
    
    #train the neural network
    def train():
        pass

    #query the neural network
    def query():
        pass
#创建一个小型神经网络，每层三个节点，学习率为0.3
input_nodes = 3
hidden_nodes = 3
output_nodes = 3
learning_rate = 0.3
#创建一个对象
n = neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

