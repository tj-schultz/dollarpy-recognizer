"""
name: path.py -- dollargeneral-recognizer
description: Path and point object definitions
authors: TJ Schultz, Skylar McCain
date: 4/4/22
"""
from point import Point


    def __str__(self):
        return "({},{})".format(int(self.x), int(self.y))

    def __repr__(self):
        return "({},{})".format(int(self.x), int(self.y))

## N_Path class
class N_Path():
    #stores the each uni-stroke of the multistoke gesture (N-strokes) as a separate list element
    strokes = None
    # n stores the number of stroke in the multi-strokest gesture == |stokes|
    n = 0
    # ?? Path variable to store the entire path of the multistroke in a single path -- stich together as strokes are added 

    def __init__(self, p=None):
        self.strokes = []
        if type(p) == Path: 
            self.stitch(p)
    
    def stitch(self, p):
        self.strokes.append(p)
        n = len(self.strokes)

    def get_n(self):
        return self.n
    
    #return the entire length of the multi-stoke path as a unitstroke
    def __len__(self):
        return 
    
    # def heap_permute(self, order, orders):
    #     if self.n == 1:
    #        orders.append(order)
    #     else:
    #         for i in range(0,self.n):
    #             self.heap_permute(self.n-1, order, orders)
    #             if(self.n%2 != 0):
    #                 temp = order[0]
    #                 order[0] = order[self.n-1]
    #                 order[self.n-1] = temp
    #             else:
    #                 temp = order[i]
    #                 order[i] = order[self.n-1]
    #                 order[self.n-1] = temp
            

## path class
class Path():
    parsed_path = None

    def __init__(self, p=None):

        self.parsed_path = []
        if type(p) == Point:    ## p is a singular starting point
            self.stitch(p)
        elif type(p) == list:   ## p is a list of Points
            for _p in p:
                self.stitch(_p)


    def __len__(self):
        return len(self.parsed_path) if self.parsed_path is not None else 0

    def __str__(self):
        path_str = 'Path length: %s\tPath:\n' % len(self)
        for p in self.parsed_path:
            path_str += ('->(%s,%s)' % (p.x, p.y))
        return path_str

    ## appends point to end of path
    def stitch(self, p):
        self.parsed_path.append(p)

    ## inserts point at index i
    def insert(self, i, p):
        if i < len(self.parsed_path):
            self.parsed_path.insert(i, p)