import random
import config

def star_graph_maker(numOfPathsFromSource,lenOfEachPath, maxNodes, reverse=False):
    # numOfNodes = numOfPathsFromSource * (lenOfEachPath - 1) + 1
    nodes = list(range(maxNodes))
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

def generate_and_save_data(numOfSamples, numOfPathsFromSource, lenOfEachPath, maxNodes, reverse=False, showLoadingBar = True):
    with open('data.txt', 'w') as file:
        for x in range(numOfSamples):
            random.seed(x)
            edgeList, path, source, goal = star_graph_maker(numOfPathsFromSource,lenOfEachPath, maxNodes, reverse)
            file.write("|".join([",".join(str(i) for i in x) for x in edgeList])+f"/{source},{goal}={','.join([str(i) for i in path])}\n")
            # loading bar
            if showLoadingBar:
                numberOfRectangles = int((x+1)*50/numOfSamples)
                bar = '█'*numberOfRectangles + " "*(50-numberOfRectangles)
                print(f'\r|{bar}| {(x+1)*100/numOfSamples:.1f}%', end="", flush=True)




if __name__ == "__main__":
    generate_and_save_data(config.numOfSamples, config.numOfPathsFromSource, config.lenOfEachPath, config.maxNodes, config.reverse, config.showLoadingBar)
