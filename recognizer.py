"""
name: recognizer.py -- dollargeneral-recognizer
description: Recognizer class with member functions to resample, rotate, scale/translate paths and
run calculations to determine the score for a particular recognizer
authors: TJ Schultz, Skylar McCain
date: 1/27/22
"""
import math
from math import acos, atan, atan2, pi, sin, cos, sqrt, dist, radians, inf
import path as pth
import dollar
import numpy as np
from time import time 

## recognizer class containing canvas display methods for
class Recognizer():
    preprocessed = {}
    use_protractor = False
    ## preprocess the template set on init
    def __init__(self, template_dict={}, protractor=False):

        self.use_protractor = protractor

        ## recursively call preproccess to build the sub-dictionary
        self.preprocessed = self.recursive_preprocess(template_dict)


    ## returns distance between points in non-pixel units
    def distance(self, p1, p2):
        return math.sqrt(math.pow(p2.x - p1.x, 2) + math.pow(p2.y - p1.y, 2))

    ## gets the path length as a sum of distances
    def path_length(self, path):
        d = 0

        ## add distances
        for i in range(len(path) - 1):
            d = d + self.distance(path.parsed_path[i], path.parsed_path[i + 1])

        return d

    ## resamples a path into n evenly spaced points
    def resample(self, path, n):
        if n <= 1:
            print("n <= 1 in resample")
            return path

        interval = self.path_length(path) / (n - 1)
        ## fix undershot paths
        interval *= 0.975
        dist = 0

        ## create a copy path
        copy = path

        ## create return path
        new_path = pth.Path()

        i = 0
        while i < len(copy.parsed_path)-1:
            #print(i, len(copy), len(new_path))

            p = copy.parsed_path[i]
            q = copy.parsed_path[i+1]

            ## calc distance
            d = self.distance(p, q)
            if dist + d > interval:

                ## interpolate new values
                qx = p.x + (((interval - dist) / d) * (q.x - p.x))
                qy = p.y + (((interval - dist) / d) * (q.y - p.y))
                q = pth.Point(qx, qy)

                ## stitch point to new path and copy point to copy path
                new_path.stitch(q)
                copy.insert(i + 1, q)

                ## reset dist
                dist = 0
            else:
                ## add distance
                dist = dist + d
            i = i + 1
        
        return new_path

    ## returns tuple of point coordinate mins and max
    def bbox(self, path):
        x_min, x_max, y_min, y_max = (math.inf, 0, math.inf, 0)

        for p in path.parsed_path:
            if p.x <= x_min:
                x_min = p.x
            if p.x > x_max:
                x_max = p.x
            if p.y <= y_min:
                y_min = p.y
            if p.y > y_max:
                y_max = p.y
        return (x_min, x_max, y_min, y_max)

    def centroid(self, path):
        (x_min, x_max, y_min, y_max) = self.bbox(path)
        x = x_min + ((x_max - x_min) / 2.0)
        y = y_min + ((y_max - y_min) / 2.0)
        return pth.Point(x, y)

    ## rotates path by theta
    def rotate_by(self, path, theta):

        rotated = pth.Path()

        ## calc centroid
        cent = self.centroid(path)

        ## perform rotation for each point
        for p in path.parsed_path:
            qx = ((p.x - cent.x) * math.cos(theta)) - \
                 ((p.y - cent.y) * math.sin(theta)) + cent.x
            qy = ((p.x - cent.x) * math.sin(theta)) + \
                 ((p.y - cent.y) * math.cos(theta)) + cent.y
            rotated.stitch(pth.Point(qx, qy))

        return rotated

    ## rotates the points so their indicative angle is 0 degrees
    def rotate_to_zero(self, path):
        if(len(path.parsed_path) == 0):
            return path
        cent = self.centroid(path)
        theta = math.atan2((cent.y - path.parsed_path[0].y), (cent.x - path.parsed_path[0].x))
        new_path = self.rotate_by(path, (theta * -1.0))
        return new_path

    ## scales points to square aspect ratio
    def scale_to_square(self, path, size):

        ## get bbox info
        bbox = self.bbox(path)
        b_width = float(bbox[1]) - float(bbox[0])
        b_height = float(bbox[3]) - float(bbox[2])

        new_path = pth.Path()

        if(b_height > 0 and b_width > 0):
            for p in path.parsed_path:
                qx = p.x * (size / b_width)
                qy = p.y * (size / b_height)
                new_path.stitch(pth.Point(qx, qy))

        return new_path

    ## translates points around the origin
    def translate_to_origin(self, path):
        cent = self.centroid(path)

        new_path = pth.Path()

        for p in path.parsed_path:
            qx = p.x - cent.x
            qy = p.y - cent.y
            new_path.stitch(pth.Point(qx, qy))

        return new_path

    ## sum and average distance between two point paths
    def path_distance(self, A, B):
        d = 0
        for i in range(min(len(A), len(B))):
            d = d + self.distance(A.parsed_path[i], B.parsed_path[i])
        if(min(len(A), len(B)) == 0):
            if(max(len(A), len(B)) == 0):
                return d
            else:
                return d / max(len(A), len(B))
        return d / min(len(A), len(B))

    ## distance at angle
    def distance_at_angle(self, path, template, theta):
        new_path = self.rotate_by(path, theta)
        d = self.path_distance(new_path, template)
        return d


    ## distance at best angle with default parameters for theta a, b and delta
    def distance_best_angle(self, path, template, atheta=-45, btheta=45, delta=2):

        ## calculate golden const
        PHI = round(0.5 * (-1.0 + math.sqrt(5)), 5)

        ## calculated variables
        x1 = (PHI * atheta) + ((1.0 - PHI) * btheta)
        f1 = self.distance_at_angle(path, template, x1)

        x2 = ((1.0 - PHI) * atheta) + (PHI * btheta)
        f2 = self.distance_at_angle(path, template, x2)

        ## find the optimum angle using delta
        while btheta - atheta > delta:
            if f1 < f2:
                btheta = x2
                x2 = x1
                f2 = f1
                x1 = (PHI * atheta) + ((1.0 - PHI) * btheta)
                f1 = self.distance_at_angle(path, template, x1)
            else:
                atheta = x1
                x1 = x2
                f1 = f2
                x2 = ((1.0 - PHI) * atheta) + (PHI * btheta)
                f2 = self.distance_at_angle(path, template, x2)

        ## return the minimum distance from f1, f2
        return min(f1, f2)


    ## create a normalized vector object of length 2n from a path
    def vectorize(self, path, o_sensitive):
        centered = self.translate_to_origin(path)
        theta = math.atan2(path.parsed_path[0].y, path.parsed_path[0].x)
        delta = 0
        if o_sensitive:
            base_orientation = (math.pi / 4.0) *\
                               math.floor((theta + (math.pi / 8.0)))
            delta = base_orientation - theta
        else:
            delta = -1.0 * theta
        sum = 0
        vector = []
        for p in centered.parsed_path:
            ## find and sum new x and y components to the vector
            qx = p.x * math.cos(delta) - p.y * math.sin(delta)
            qy = p.y * math.cos(delta) + p.x * math.sin(delta)
            vector.append(qx)
            vector.append(qy)

            ## add the sum for this point
            sum = sum + (qx * qx) + (qy * qy)

        ## normalize
        magnitude = math.sqrt(sum)
        for i in range(len(vector)):
            vector[i] = vector[i] / magnitude

        if len(vector) < 0:
            print("vector < 0")
        return vector

    ## optimal cosine distance function to calculate the OCD for two vectors
    def opt_cos_distance(self, u, v):
        a = 0
        b = 0
        for i in range(0, len(u), 2):
            a = a + (u[i] * v[i]) + (u[i + 1] * v[i + 1])
            b = b + (u[i] * v[i + 1]) - (v[i] * u[i + 1])
        theta = math.atan(b / a)
        return math.acos(a * math.cos(theta) + b * math.sin(theta))


    ## preprocess path to compare
    def preprocess(self, path):
        ## resample the points
        new_path = self.resample(path, dollar.Dollar.prefs["n_points"])
       
        ## performing protractor preprocessing
        if self.use_protractor:
            return self.vectorize(new_path, False)

        ## rotate to indicative angle
        new_path = self.rotate_to_zero(new_path)

        ## scale to size box
        new_path = self.scale_to_square(new_path, dollar.Dollar.prefs["square_size"])

        ## translate to origin
        new_path = self.translate_to_origin(new_path)

        return new_path

    ## recursive preprocessing function for path dictionaries
    def recursive_preprocess(self, template_dict={}):
        for k, v in template_dict.items():
            if isinstance(v, dict):
                ## recursively call constructor to build the sub-dictionary
                template_dict[k] = self.recursive_preprocess(template_dict[k])
            else:
                ## preprocess and replace the Path object
                new_path = self.preprocess(template_dict[k])
                template_dict[k] = new_path

        ## copy template dictionary at next highest level of recursion
        return template_dict

    ## recognizer method -- combines steps in performing scoring and can alternatively be
    def recognize(self, path, templates={}, preprocess=False):

        ## if no specified template dict, set to preprocessed dictionary formed at instantiation
        if templates == {}:
            templates = self.preprocessed

        ## scores array
        scores = []
        if len(path) < 1:
            return

        ## preprocess the candidate path into a Path object
        candidate = path
        if preprocess:
            candidate = self.preprocess(path)


        ## if recognizing according to protractor
        if self.use_protractor:
            max = 0
            for t_key in templates.keys():
                d = self.opt_cos_distance(templates[t_key], candidate)
                dscore = 1.0 / d
                scores.append((t_key, dscore))
                if dscore > max:
                    max = dscore
            score = max
        else:
            ## for each preprocessed template, compare the path and calculate the max score
            b = (0.5 * math.sqrt(2.0 * math.pow(dollar.Dollar.prefs["square_size"], 2)))

            hd = (0.5 * math.sqrt(2.0 * math.pow(dollar.Dollar.prefs["square_size"], 2)))
            for t_key in templates.keys():
                ## get distance
                d = self.distance_best_angle(candidate, templates[t_key])

                ## calculate score
                dscore = 1.0 - (d / hd)
                scores.append((t_key, dscore))
        scores.sort(key=lambda y: y[1], reverse=True)

        #print(scores)
        return scores

class Point():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return "({},{})".format(self.x, self.y)

    def __repr__(self):
        return "({},{})".format(self.x, self.y)

#  NDollarRecognizer constants
NumMultistrokes = 16
NumPoints = 96
SquareSize = 250.0
# customize to desired gesture set (usually 0.20 - 0.35)
OneDThreshold = 0.25;
Origin = Point(0,0)
Diagonal = sqrt(SquareSize * SquareSize + SquareSize * SquareSize);
HalfDiagonal = 0.5 * Diagonal
AngleRange = radians(45.0)
AnglePrecision = radians(2.0)
# Golden Ratio
Phi = 0.5 * (-1.0 + sqrt(5.0))
# eighth of gesture length
StartAngleIndex = (NumPoints / 8); 
AngleSimilarityThreshold = radians(30.0)

class Rectangle():
    def __init__(self, x, y, width, height):
        self.x = x;
        self.y = y;
        self.Width = width if width != 0 else 0.000001;
        self.Height = height if height != 0 else 0.000001;

# Unistroke class: a unistroke template
class Unistroke():
    def __init__(self, name, useBoundedRotationInvariance, points):
        self.name = name
        self.points = Resample(points, NumPoints)
        radians = IndicativeAngle(self.points)
        self.points = RotateBy(self.points, -radians)
        self.points = ScaleDimTo(self.points, SquareSize, OneDThreshold)
        if (useBoundedRotationInvariance):
            # restore
            self.points = RotateBy(self.points, +radians)
        self.points = TranslateTo(self.points, Origin)
        self.startUnitVector = CalcStartUnitVector(self.points, StartAngleIndex)
        # for Protractor
        self.vector = Vectorize(self.points, useBoundedRotationInvariance)

# Multistroke class: a container for unistrokes
class Multistroke():
    def __init__(self, name, useBoundedRotationInvariance, strokes):
        self.name = name
        print(name)
        # number of individual strokes
        self.numStrokes = len(strokes)

        # array of integer indices
        order = [i for i in range(len(strokes))]
        orders = []
        # array of integer arrays
        HeapPermute(len(strokes), order, orders)

        # returns array of point arrays
        unistrokes = MakeUnistrokes(strokes, orders)
        # unistrokes for this multistroke
        self.Unistrokes = [Unistroke(name, useBoundedRotationInvariance, unistroke) for unistroke in unistrokes]

class Result():
    def __init__(self, name, score, ms):
        self.name = name;
        self.score = score;
        self.time = ms;


class NDollarRecognizer():
    def __init__(self, useBoundedRotationInvariance=False):
        # one predefined multistroke for each multistroke type
        self.useBoundedRotationInvariance = useBoundedRotationInvariance
        self.Multistrokes = [
            Multistroke("T", useBoundedRotationInvariance, [
                [Point(30,7),Point(103,7)],
                [Point(66,7),Point(66,87)]
            ]),
            Multistroke("N", useBoundedRotationInvariance, [
                [Point(177,92),Point(177,2)],
                [Point(182,1),Point(246,95)],
                [Point(247,87),Point(247,1)]
            ]),
            Multistroke("D", useBoundedRotationInvariance, [
                [Point(345,9),Point(345,87)],
                [Point(351,8),Point(363,8),Point(372,9),Point(380,11),Point(386,14),Point(391,17),Point(394,22),Point(397,28),Point(399,34),Point(400,42),Point(400,50),Point(400,56),Point(399,61),Point(397,66),Point(394,70),Point(391,74),Point(386,78),Point(382,81),Point(377,83),Point(372,85),Point(367,87),Point(360,87),Point(355,88),Point(349,87)]
            ]),
            Multistroke("P", useBoundedRotationInvariance, [
                [Point(507,8),Point(507,87)],
                [Point(513,7),Point(528,7),Point(537,8),Point(544,10),Point(550,12),Point(555,15),Point(558,18),Point(560,22),Point(561,27),Point(562,33),Point(561,37),Point(559,42),Point(556,45),Point(550,48),Point(544,51),Point(538,53),Point(532,54),Point(525,55),Point(519,55),Point(513,55),Point(510,55)]
            ]),
            Multistroke("X", useBoundedRotationInvariance, [
                [Point(30,146),Point(106,222)],
                [Point(30,225),Point(106,146)]
            ]),
            Multistroke("H", useBoundedRotationInvariance, [
                [Point(188,137),Point(188,225)],
                [Point(188,180),Point(241,180)],
                [Point(241,137),Point(241,225)]
            ]),
            Multistroke("I", useBoundedRotationInvariance, [
                [Point(371,149),Point(371,221)],
                [Point(341,149),Point(401,149)],
                [Point(341,221),Point(401,221)]
            ]),
            Multistroke("exclamation", useBoundedRotationInvariance, [
                [Point(526,142),Point(526,204)],
                [Point(526,221)]
            ]),
            Multistroke("line", useBoundedRotationInvariance, [
                [Point(12,347),Point(119,347)]
            ]),
            Multistroke("five-point star", useBoundedRotationInvariance, [
                [Point(177,396),Point(223,299),Point(262,396),Point(168,332),Point(278,332),Point(184,397)]
            ]),
            Multistroke("null", useBoundedRotationInvariance, [
                [Point(382,310),Point(377,308),Point(373,307),Point(366,307),Point(360,310),Point(356,313),Point(353,316),Point(349,321),Point(347,326),Point(344,331),Point(342,337),Point(341,343),Point(341,350),Point(341,358),Point(342,362),Point(344,366),Point(347,370),Point(351,374),Point(356,379),Point(361,382),Point(368,385),Point(374,387),Point(381,387),Point(390,387),Point(397,385),Point(404,382),Point(408,378),Point(412,373),Point(416,367),Point(418,361),Point(419,353),Point(418,346),Point(417,341),Point(416,336),Point(413,331),Point(410,326),Point(404,320),Point(400,317),Point(393,313),Point(392,312)],
                [Point(418,309),Point(337,390)]
            ]),
            Multistroke("arrowhead", useBoundedRotationInvariance, [
                [Point(506,349),Point(574,349)],
                [Point(525,306),Point(584,349),Point(525,388)]
            ]),
            Multistroke("pitchfork", useBoundedRotationInvariance, [
                [Point(38,470),Point(36,476),Point(36,482),Point(37,489),Point(39,496),Point(42,500),Point(46,503),Point(50,507),Point(56,509),Point(63,509),Point(70,508),Point(75,506),Point(79,503),Point(82,499),Point(85,493),Point(87,487),Point(88,480),Point(88,474),Point(87,468)],
                [Point(62,464),Point(62,571)]
            ]),
            Multistroke("six-point star", useBoundedRotationInvariance, [
                [Point(177,554),Point(223,476),Point(268,554),Point(183,554)],
                [Point(177,490),Point(223,568),Point(268,490),Point(183,490)]
            ]),
            Multistroke("asterisk", useBoundedRotationInvariance, [
                [Point(325,499),Point(417,557)],
                [Point(417,499),Point(325,557)],
                [Point(371,486),Point(371,571)]
            ]),
            Multistroke("half-note", useBoundedRotationInvariance, [
                [Point(546,465),Point(546,531)],
                [Point(540,530),Point(536,529),Point(533,528),Point(529,529),Point(524,530),Point(520,532),Point(515,535),Point(511,539),Point(508,545),Point(506,548),Point(506,554),Point(509,558),Point(512,561),Point(517,564),Point(521,564),Point(527,563),Point(531,560),Point(535,557),Point(538,553),Point(542,548),Point(544,544),Point(546,540),Point(546,536)]
            ])
        ]
        self.NumMultistrokes = len(self.Multistrokes)

    #
    # The $N Gesture Recognizer API begins here -- 3 methods: Recognize(), AddGesture(), and DeleteUserGestures()
    #
    def Recognize(self, strokes, useBoundedRotationInvariance, requireSameNoOfStrokes, useProtractor):
        if len(strokes) == 0:
            return Result("Null", 0.0, 0)
        t0 = time()
        points = CombineStrokes(strokes) # make one connected unistroke from the given strokes
        candidate = Unistroke("", useBoundedRotationInvariance, points)
        u = -1
        b = inf
        # for each multistroke template
        for multistroke in self.Multistrokes:
            # optional -- only attempt match when same # of component strokes
            if not requireSameNoOfStrokes or len(strokes) == self.Multistrokes[i].NumStrokes:
                # for each unistroke within this multistroke
                for unistroke in multistroke:
                    # strokes start in the same direction
                    if AngleBetweenUnitVectors(candidate.StartUnitVector, self.Multistrokes[i].Unistrokes[j].StartUnitVector) <= AngleSimilarityThreshold:
                        d = 0
                        if (useProtractor):
                            # Protractor
                            d = OptimalCosineDistance(self.Multistrokes[i].Unistrokes[j].Vector, candidate.Vector) 
                        else:
                            # Golden Section Search (original $N)
                            d = DistanceAtBestAngle(candidate.Points, self.Multistrokes[i].Unistrokes[j], -AngleRange, +AngleRange, AnglePrecision) 
                        if (d < b):
                            # best (least) distance
                            b = d 
                            # multistroke owner of unistroke
                            u = i 
        t1 = time()
        return (u == -1) if Result("No match.", 0.0, t1-t0) else Result(self.Multistrokes[u].Name, useProtractor if (1.0 - b) else (1.0 - b / HalfDiagonal), t1-t0)
    
    def AddGesture(name, useBoundedRotationInvariance, strokes):
        self.Multistrokes[len(self.Multistrokes)] = Multistroke(name, useBoundedRotationInvariance, strokes)
        num = 0
        for multistroke in self.Multistrokes:
            if multistroke.Name == name:
                num += 1
        return num
    
    def DeleteUserGestures():
        pass

#
# Private helper functions from here on down
#
def HeapPermute(n, order, orders):
    if (n == 1):
        # append copy
        orders.append(order)
    else:
        for i in range(n):
            HeapPermute(n - 1, order, orders)
            if (n % 2 == 1):
                # swap 0, n-1
                order[0], order[n-1] = order[n-1], order[0]
            else:
                # swap i, n-1
                order[i], order[n-1] = order[n-1], order[i]

def MakeUnistrokes(strokes, orders):
    # array of point arrays
    unistrokes = [] 
    for order in orders:
        # use b's bits for directions
        for b in range(pow(2, len(order))):
            # array of points
            unistroke = [] 
            for i in range(len(order)):
                pts = []
                # is b's bit at index i on?
                if (((b >> i) & 1) == 1):
                    # copy and reverse
                    pts = strokes[order[i]][::-1]
                else:
                    # copy
                    pts = strokes[order[i]]
                # append points
                unistroke.append(pts)
            # add one unistroke to set
            for u in unistroke:
                unistrokes.append(u)
    return unistrokes

def CombineStrokes(strokes):
    return [Point(p.x, p.y) for s in strokes for p in s]

def Resample(points, n):
    D = 0.0
    # interval length
    I = PathLength(points) / (n - 1)
    newpoints = [points[0]]
    i = 1
    while True:
        if i == 1:
            if len(points) == 1:
                return newpoints
        d = Distance(points[i-1], points[i])
        if ((D + d) >= I):
            q = Point(points[i-1].x + ((I - D) / d) * (points[i].x - points[i-1].x), 
                      points[i-1].y + ((I - D) / d) * (points[i].y - points[i-1].y))
            # append new point 'q'
            newpoints.append(q)
            # insert 'q' at position i in points s.t. 'q' will be the next i
            points.insert(i, q) 
            D = 0.0
        else:
            D += d
        i += 1
        if i == len(points):
            break
    # somtimes we fall a rounding-error short of adding the last point, so add it if so
    if len(newpoints) == n - 1:
        newpoints.append(Point(points[-1].x, points[-1].y))
    return newpoints


def IndicativeAngle(points):
    c = Centroid(points)
    return atan2(c.y - points[0].y, c.x - points[0].x)


# rotates points around centroid
def RotateBy(points, radians):
    # c = Centroid(points)
    # cos_r = cos(radians)
    # sin_r = sin(radians)
    # newpoints = []
    # for p in points:
    #     qx = (p.x - c.x) * cos_r - (p.y - c.y) * sin_r + c.x
    #     qy = (p.x - c.x) * sin_r + (p.y - c.y) * cos_r + c.y
    #     newpoints[-1] = (qx, qy)
    # return newpoints

    c = Centroid(points)
    cos_r = cos(radians)
    sin_r = sin(radians)
    newpoints = [Point((p.x - c.x) * cos_r - (p.y - c.y) * sin_r + c.x, 
                       (p.x - c.x) * sin_r + (p.y - c.y) * cos_r + c.y) for p in points]
    return newpoints


# scales bbox uniformly for 1D, non-uniformly for 2D
def ScaleDimTo(points, size, ratio1D):
    B = BoundingBox(points)
    uniformly = min(B.Width / B.Height, B.Height / B.Width) <= ratio1D # 1D or 2D gesture test
    newpoints = [Point(p.x * (size / max(B.Width, B.Height)) if uniformly else p.x * (size / B.Width), 
                       p.y * (size / max(B.Width, B.Height)) if uniformly else p.y * (size / B.Height)) for p in points]
    return newpoints


# translates points' centroid
def TranslateTo(points, pt):
    c = Centroid(points)
    newpoints = [Point(p.x + pt.x - c.x,
                       p.y + pt.y - c.y) for p in points]
    return newpoints


# for Protractor
def Vectorize(points, useBoundedRotationInvariance):
    cos = 1.0;
    sin = 0.0;
    if (useBoundedRotationInvariance):
        iAngle = atan2(points[0].y, points[0].x)
        baseOrientation = (pi / 4.0) * floor((iAngle + pi / 8.0) / (pi / 4.0))
        cos = cos(baseOrientation - iAngle)
        sin = sin(baseOrientation - iAngle)
    sum = 0.0
    vector = []
    for p in points:
        newX = p.x * cos - p.y * sin
        newY = p.y * cos + p.x * sin
        vector.append(newX)
        vector.append(newY)
        sum += newX * newX + newY * newY
    magnitude = sqrt(sum)
    if magnitude == 0:
        return Origin
    for v in vector:
        v = v / magnitude
    return vector


# for Protractor
def OptimalCosineDistance(v1, v2):
    a = 0.0;
    b = 0.0;
    for i in range(0, len(v1), 2):
        a += v1[i] * v2[i] + v1[i+1] * v2[i+1];
        b += v1[i] * v2[i+1] - v1[i+1] * v2[i];
    angle = atan(b / a)
    return acos(a * cos(angle) + b * sin(angle))


def DistanceAtBestAngle(points, T, a, b, threshold):
    x1 = Phi * a + (1.0 - Phi) * b
    f1 = DistanceAtAngle(points, T, x1)
    x2 = (1.0 - Phi) * a + Phi * b
    f2 = DistanceAtAngle(points, T, x2)
    while (abs(b - a) > threshold):
        if (f1 < f2):
            b = x2
            x2 = x1
            f2 = f1
            x1 = Phi * a + (1.0 - Phi) * b
            f1 = DistanceAtAngle(points, T, x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = (1.0 - Phi) * a + Phi * b
            f2 = DistanceAtAngle(points, T, x2)
    return min(f1, f2)


# average distance between corresponding points in two paths
def PathDistance(pts1, pts2):
    d = 0.0;
    # assumes pts1.length == pts2.length
    for p1, p2 in map(pts1, pts2):
        d += Distance(p1, p2)
    return d / len(pts1)


def DistanceAtAngle(points, T, radians):
    newpoints = RotateBy(points, radians)
    return PathDistance(newpoints, T.Points)


# def centeroid(points):
#     length = arr.shape[0]
#     sum_x = np.sum(arr[:, 0])
#     sum_y = np.sum(arr[:, 1])
#     return (sum_x / length, sum_y / length)
def Centroid(points):
    x, y = 0.0, 0.0
    for p in points:
        x += p.x
        y += p.y
    x /= len(points)
    y /= len(points)
    return Point(x, y)


# def bounding_box(points):
#     pts = [(p.x, p.y) for p in points]
#     min_x, min_y = numpy.min(pts[0], axis=0)
#     max_x, max_y = numpy.max(pts[0], axis=0)
#     return numpy.array([(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)])
def BoundingBox(points):
    minX = inf
    maxX = -inf
    minY = inf
    maxY = -inf
    for p in points:
        minX = min(minX, p.x)
        minY = min(minY, p.y)
        maxX = max(maxX, p.x)
        maxY = max(maxY, p.y)
    return Rectangle(minX, minY, maxX - minX, maxY - minY)

# length traversed by a point path
# def PathLength(points):
#     d = 0.0
#     apts = np.array([(p.x, p.y) for p in points])
#     lengths = np.sqrt(np.sum(np.diff(apts, axis=0)**2, axis=1))
#     return np.sum(lengths)
def PathLength(points):
    d = 0.0
    for i in range(1, len(points)):
        d += Distance(points[i - 1], points[i])
    return d

# distance between two points
def Distance(p1, p2):
    return dist([p1.x, p1.y], [p2.x, p2.y])

# start angle from points[0] to points[index] normalized as a unit vector
def CalcStartUnitVector(points, index):
    index = int(index)
    v = Point(points[index].x - points[0].x, points[index].y - points[0].y) if len(points) != 1 else points[0]
    length = sqrt(v.x * v.x + v.y * v.y)
    if length == 0:
        return Origin
    return Point(v.x / length, v.y / length)

#  gives acute angle between unit vectors from (0,0) to v1, and (0,0) to v2
def AngleBetweenUnitVectors(v1, v2):
    n = (v1.x * v2.x + v1.y * v2.y)
    #  ensure [-1,+1]
    c = max(-1.0, min(1.0, n))
    # arc cosine of the vector dot product
    return acos(c)