## point class
class Point():
    ## point coordinates
    x = 0
    y = 0

    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
    def __str__(self):
        return "({},{})".format(self.x, self.y)

    def __repr__(self):
        return "({},{})".format(self.x, self.y)