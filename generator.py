import random
import csv

def star_graph_maker(numOfPathsFromSource,lenOfEachPath, reverse=False):
    numOfNodes = numOfPathsFromSource * (lenOfEachPath - 1) + 1
    nodes = list(range(numOfNodes))
    random.shuffle(nodes)
    
    source = nodes.pop()
    
    edgeList = []
    path = [source]
    
    for p in range(numOfPathsFromSource):
        oldNode = source
        for i in range(lenOfEachPath - 1):
            newNode = nodes.pop()
            edgeList.append((oldNode, newNode))
            oldNode = newNode
            if p == 0:
                path.append(oldNode)
        if p == 0:
            goal = oldNode
    random.shuffle(edgeList)
    
    if reverse:
        path = path[::-1]
    
    return edgeList, path, source, goal
        

# print(star_graph_maker(4, 4))

def generate_and_save_data(numOfSamples, numOfPathsFromSource,lenOfEachPath, reverse=False, showLoadingBar = True):
    with open('data.csv', 'w', newline="") as file:
        writer = csv.writer(file)
        
        writer.writerow(["edgeList", "path", "source", "goal"])
        for x in range(numOfSamples):
            random.seed(x)
            edgeList, path, source, goal = star_graph_maker(numOfPathsFromSource,lenOfEachPath, reverse)
            writer.writerow([edgeList, path, source, goal])
            # loading bar
            if showLoadingBar:
                numberOfRectangles = int((x+1)*50/numOfSamples)
                bar = '█'*numberOfRectangles + " "*(50-numberOfRectangles)
                print(f'\r|{bar}| {(x+1)*100/numOfSamples:.1f}%', end="", flush=True)

generate_and_save_data(10000, 4, 4)
