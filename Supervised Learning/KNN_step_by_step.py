import csv
import random

def loadDataSet (filename, split, trainingSet= [], testSet=[]):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)-1):
            for y in range(4):
                dataset[x][y]= float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])

# Id,SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm,Species
trainingSet =[]
testSet =[]
loadDataSet(r'C:/Users/MahaRishi/Documents/machine learning from scratch/datasets/Iris.csv', 0.66, trainingSet, testSet)
print('Train : ', len(trainingSet))
print('Test : ', len(testSet))

# STEP 2 - SIMILARITY
import math
def euclideanDistance(instance1, instance2, length):
    distance =0
    for x in range(length):
        distance+= pow((float(instance1[x])- float(instance2[x])), 2)
    return math.sqrt(distance)

data1= [2,2,2, 'a']
data2= [4,4,4, 'b']
distance = euclideanDistance(data1, data2, 3)
print('Distance : '+repr(distance))

# STEP 3 - Neighbours
def getNeighbours(trainingSet, testInstance, k):
    distances = []
    length= len(testInstance)-1
    for x in range(len(trainingSet)):
        dist= euclideanDistance(testInstance, trainingSet[x],length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=lambda elem: elem[1])
    # distances.sort(key=operator.itemgetter(1))
    neighbours =[]
    for x in range(k):
        neighbours.append(distances[x][0])
    return neighbours

trainSet = [[2,2,2, 'a'], [4,4,4, 'b']]
testInstance=[5,5,5]
k=1
neighbours = getNeighbours(trainSet, testInstance, 1)
print(neighbours)

# STEP 4 - Response 
def getResponse(neighbours):
    classVotes = {}
    for x in range(len(neighbours)):
        response = neighbours[x][-1]
        if response in classVotes:
            classVotes[response]+=1
        else:
            classVotes[response]=1
        sortedVotes = sorted (classVotes.items(), key=lambda elem: elem[1], reverse = True)
        # print(sortedVotes)
        return sortedVotes[0][0]

neighbours = [[1,1,1,'a'],[2,2,2,'a'], [3,3,3,'b']]
response = getResponse(neighbours)
print(response)

# STEP 5 - Accuracy
def getAccuracy(testSet, predictions):
    correct = 0
    for x in range (len(testSet)):
        if testSet[x][-1] is predictions[x]:
            print(correct)
            correct +=1
    return (correct/ float(len(testSet)))*100

testSet = [[1,1,1,'a'], [2,2,2,'a'],[3,3,3,'b']]
predictions = ['a','a','a']
accuracy = getAccuracy(testSet, predictions)
print(accuracy)

# LAST STEP - Putting all our functions together and IRIS dataset
def main():
    # prepare data
    trainingSet=[]
    testSet=[]
    split = 0.67
    loadDataSet('C:/Users/MahaRishi/Documents/machine learning from scratch/datasets/Iris.csv', split, trainingSet, testSet)
    print('Train set : ', len(trainingSet))
    print('Test set : ', len(testSet))
    # generate predictions
    predictions =[]
    k = 3
    for x in range(len(testSet)):
        neighbours= getNeighbours(trainingSet, testSet[x], k)
        result= getResponse(neighbours)
        predictions.append(result)
        print('>predicted= '+repr(result)+ 'actual= '+repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy : ',accuracy, "%")

main()