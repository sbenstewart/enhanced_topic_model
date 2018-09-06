import Tokener
from sparseAutoencoder import SparseAutoencoder
import numpy as np
import math
import time
import scipy
import os

class AE(object):

    """This class implements Stacked Auto Encoder with Greedy Layer wise training
    Algorithm."""

    def __init__(self,
        input=None,         # input to AE
        hid1=140,           # First hidden layer
        hid2=40,            # Second hidden layer
        hid3=30,            # Third hidden layer
        code=10,            # The Code layer
        lamda=0.0001,       # Decay Parameter
        rho=0.01,           # Sparcity Parameter
        beta=3,             # Sparcity Penalty Term
        max_iterations=100):

        """Initialization in Auto Encoder"""

        if input is None:
            raise Exception("Please give some input")
        else:
            self.input = input

        self.lamda = lamda
        self.rho = rho
        self.beta = beta

        self.hid1 = hid1
        self.hid2 = hid2
        self.hid3 = hid3
        self.code = code

        self.max_iterations = max_iterations

        #hoo print("Model Initialized")
    def train(self):

        """Training the AE with greedy Layer Wise Algorithm"""

        # Training Outer Most Encoder
        encoderOne = SparseAutoencoder(self.input.shape[0], self.hid1, self.rho, self.lamda, self.beta)
        opt_solution  = scipy.optimize.minimize(encoderOne.sparseAutoencoderCost, encoderOne.theta, 
            args = (self.input,), method = 'L-BFGS-B', 
            jac = True, options = {'maxiter': self.max_iterations})
        opt_theta     = opt_solution.x
        self.opt_W1   = opt_theta[encoderOne.limit0 : encoderOne.limit1].reshape(self.hid1, self.input.shape[0])
    
        activationEncoderOne = np.dot(self.opt_W1, self.input)
        
		#print("Encoder One Trained")
        
		#print(activationEncoderOne.shape)

        # Training Second Outer Most Encoder
        encoderTwo = SparseAutoencoder(self.hid1, self.hid2, self.rho, self.lamda, self.beta)
        opt_solution = scipy.optimize.minimize(encoderTwo.sparseAutoencoderCost, encoderTwo.theta, 
            args = (activationEncoderOne,), method = 'L-BFGS-B', 
            jac = True, options = {'maxiter': self.max_iterations})
        opt_theta     = opt_solution.x
        self.opt_W2   = opt_theta[encoderTwo.limit0 : encoderTwo.limit1].reshape(self.hid2, self.hid1)

        activationEncoderTwo = np.dot(self.opt_W2, activationEncoderOne)
        #print("Encoder Two Tranined")
        #print(activationEncoderTwo.shape)

        # Training Third Outer Most Encoder
        encoderThree = SparseAutoencoder(self.hid2, self.hid3, self.rho, self.lamda, self.beta)
        opt_solution = scipy.optimize.minimize(encoderThree.sparseAutoencoderCost, encoderThree.theta, 
            args = (activationEncoderTwo,), method = 'L-BFGS-B', 
            jac = True, options = {'maxiter': self.max_iterations})
        opt_theta     = opt_solution.x
        self.opt_W3   = opt_theta[encoderThree.limit0 : encoderThree.limit1].reshape(self.hid3, self.hid2)

        activationEncoderThree = np.dot(self.opt_W3, activationEncoderTwo)
        #print("Encoder Three Trained")
        #print(activationEncoderThree.shape)

        # Training Code Encoder
        encoderFour = SparseAutoencoder(self.hid3, self.code, self.rho, self.lamda, self.beta)
        opt_solution = scipy.optimize.minimize(encoderFour.sparseAutoencoderCost, encoderFour.theta, 
            args = (activationEncoderThree,), method = 'L-BFGS-B', 
            jac = True, options = {'maxiter': self.max_iterations})
        opt_theta     = opt_solution.x
        self.opt_W4   = opt_theta[encoderFour.limit0 : encoderFour.limit1].reshape(self.code, self.hid3)

        activationEncoderFour = np.dot(self.opt_W4, activationEncoderThree)
        self.conceptSpace = activationEncoderFour           # Save The Final ConceptSpace
        #print("Enocder Four Tranined")
        #print(activationEncoderFour.shape)

    def cosineSimilarity(self, input):

        """This methods find cosine similarity between the sentences for Ranking"""

        similarity = []

        for vector in range(self.conceptSpace.shape[1]):

            # Iterate through every test data and find similarity #

            sim = np.dot(self.conceptSpace[:,0], self.conceptSpace[:, vector]) / (np.linalg.norm(self.conceptSpace[:,0]) * np.linalg.norm(self.conceptSpace[:, vector]))

            if math.isnan(sim.item()):
                similarity.extend([0])
            else:
                similarity.extend([sim.item()])
            #print("Similarity between S1 and S" + str(vector+1) + " is " + str(sim.item()))

        # Retrun the iterable with Similarity
        return similarity

def findMetrics(conceptSpace, sentenceIndex, trueVal):

    """ Finding the metric to evaluate the generated summary
    """

    if len(sentenceIndex) == 0:
        return -1

    initVal = conceptSpace[:, sentenceIndex[0]]

    i = 1
    while i < len(sentenceIndex):
        initVal = np.sum([initVal, conceptSpace[:, sentenceIndex[i]]], axis=0)
        i = i+1

    initVal = initVal / len(sentenceIndex)

    relevancy = np.dot(initVal, trueVal) / (np.linalg.norm(initVal) * np.linalg.norm(trueVal))

    return relevancy

def computeDeviation(conceptSpace):

    """ Computing the number of sentences in the summary.
        This uses Standard Deviation on average concept
        Space
    """

    initVal = conceptSpace[:, 0]
    i = 1
    while i < conceptSpace.shape[1]:
        initVal = np.sum([initVal, conceptSpace[:, i]], axis=0)
        i = i+1

    initVal = initVal / conceptSpace.shape[1]

    stdDev = np.std(initVal) * (10 ** 11) * 4
	
	#print(str(math.ceil(stdDev)))
	
    return math.ceil(stdDev)


if __name__ == "__main__":

    """Main block to perform Auto Encoder"""

    #outfile = open('output.txt', 'w',encoding="UTF-8")
    relFixed = []
    relVariable = []
    rand = np.random.RandomState(int(time.time()))
    # An uniform value to compare to
    trueVal = np.asarray(rand.uniform(size=10))

    fileCount = 1
    print("Starting for total set")
    for (filePath, dirNames, fileNames) in os.walk("E:/BigData/Project/Run/Files_Keywords/TextFiles"):
        for file in sorted(fileNames):
            if not os.path.exists("E:\\Project\\DataSet\\RPHirsch\\Level4_DeepOutput\\" ):#+ "_".join(varName[:3]) ):
                os.makedirs("E:\\Project\\DataSet\\RPHirsch\\Level4_DeepOutput\\" ) #+ "_".join(varName[:3]) )
            outfile = open(os.path.join("E:\\Project\\DataSet\\RPHirsch\\Level4_DeepOutput" , file),"w",encoding="UTF-8")

            print("File number" + str(fileCount))
            fileCount = fileCount + 1

            absPath = os.path.join(filePath, file)
            print(absPath)
            #f = open(absPath, 'r',encoding="UTF-8")
            f = open(absPath, 'r')
            #print(f.encoding)
            #f = open(absPath, 'r',encoding="utf-8")
            #print(f.encoding)
            ################### Tokenizinge #####################
            pter = Tokener.Pyxter(f)
            data = pter.tf()

            ################# Model training ####################
            model = AE(input=data)
            model.train()
            result = model.cosineSimilarity(data)   # `result` 
			
			#print("COSINE SIMILARITY IS ")
            
			###### sentence Ranking with fixed summary size #####
            sentenceIndex = sorted(range(len(result)), key=lambda i: result[i])[-20:]
            print('\nSummary finding')
            if 0 in sentenceIndex:
                sentenceIndex.remove(0)
            print(sentenceIndex)
            pter.printSentence(sentenceIndex, outfile, file)

            ######## metrics for fixed size summary size ########
            
            relevancy = findMetrics(model.conceptSpace, sentenceIndex, trueVal)
            relFixed.append(relevancy)
            outfile.write(str(relevancy))
            outfile.write('\n\n')
            

            ###### finding the number of sentences relating to the concept of paper ########
            n = computeDeviation(model.conceptSpace)

            ############### sentence ranking with variable size summary ####################
            sentenceIndex = sorted(range(len(result)), key=lambda i: result[i])[-n:]
            print('\nSummary finding')
            if 0 in sentenceIndex:
                sentenceIndex.remove(0)
            print(sentenceIndex)
            pter.printSentence(sentenceIndex, outfile, f.name)

            #################### metrics for variable size summary #########################
            relevancy = findMetrics(model.conceptSpace, sentenceIndex, trueVal)
            relVariable.append(relevancy)
            outfile.write(str(relevancy))
            outfile.write('\n\n\n\n')
            outfile.close()
    # Prints mean for fixed summary sized summary #
    print(np.mean(relFixed))
    # Prints mean for variable summary sized summary #
    print(np.mean(relVariable))