import numpy
import scipy.special
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

        #为神经网络设置2个链接权重矩阵
        self.wih = numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))#此处wih是Weight-input_hidden的意思，指代权重
        self.who = numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))#此处who是Weight-hidden_output的意思，指代权重
        
        #在__init__函数中，定义希望使用的激活函数
        self.activation_function = lambda x: scipy.special.expit(x)#其中lambda为匿名函数
        
    #train the neural network
    def train(self,inputs_list,target_list):
        #将输入列表转化为二阶矩阵
        inputs = numpy.array(input_list,ndmin = 2).T
        targets = numpy.arry(target_list,ndmin = 2).T
        #如何进行隐藏层计算，同query函数中的一样
        hidden_inputs = numpy.dot(self,wih,inputs)
        hidden_inputs = self.activation_function(hidden_inputs)

        #如何计算最终层，同上
        final_inputs = numpy.dot(self.who,hidden_inputs)
        final_outputs = self.activation_function(final_inputs)
        
        #计算误差，误差 = 目标值target - 最终值final_outputs
        output_errors = targets - final_outputs

        #将输出时得到的误差根据权重反向分配到隐藏层，将误差进行重组
        hidden_error = numpy.dot(self.who.T , output_errors)

        #更新隐藏层和输出层之间的分配权重
        self.who+= self.lr*numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),numpy.tranpose(hidden_outputs
            ))
        
        #更新输入层和隐藏层之间的分配权重，方法同上
        self.wih+= self.lr*numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),numpy.tranpose(inputs))
        


    #query the neural network
    def query(self,inputs_list):
        #将输入列表转化为二阶矩阵
        inputs = numpy.array(inputs_list,ndmin = 2).T

        #如何计算隐藏层的信号
        #应用numpy库，将链接权重矩阵wih点乘输入矩阵I
        hidden_inputs = numpy.dot(self.wih,inputs)
        #应用scipy库导入S抑制函数之后，将激活函数应用到组合调整之后准备进入到隐藏层节点的信号
        hidden_outputs = self.activation_function(hidden_inputs)

        #如何进行最终输出层的计算，与隐藏层类似
        #应用numpy库，将链接权重矩阵wih点乘输入矩阵I
        final_inputs = numpy.dot(self.who,hidden_outputs)
        #应用scipy库导入S抑制函数之后，将激活函数应用到组合调整之后准备进入到隐藏层节点的信号
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
    
def main():
    #创建一个小型神经网络，每层三个节点，学习率为0.3
    input_nodes = 3
    hidden_nodes = 3
    output_nodes = 3
    learning_rate = 0.3
    #创建一个对象
    n = neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)


if __name__ == "__main__":
    main()
