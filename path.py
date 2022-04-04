"""
name: path.py -- dollargeneral-recognizer
description: Path and point object definitions
authors: TJ Schultz, Skylar McCain
date: 4/4/22
"""

## point class
class Point():
    ## point coordinates
    x = 0
    y = 0

    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)

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
        return len(self.parsed_path)

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