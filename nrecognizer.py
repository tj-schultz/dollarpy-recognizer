import math
from math import acos, atan, atan2, fabs, floor, sin, cos, sqrt, radians, inf
from math import pi as PI
from random import uniform
from time import time 
from itertools import permutations

class Point():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return "({},{})".format(int(self.x), int(self.y))

    def __repr__(self):
        return "({},{})".format(int(self.x), int(self.y))

#  NDollarRecognizer constants
NUM_MULTISTROKES = 16
NUM_POINTS = 96
SQUARE_SIZE = 250.0
# customize to desired gesture set (usually 0.20 - 0.35)
ONE_D_THRESHOLD = 0.25;
ORIGIN = Point(0,0)
DIAGONAL = sqrt(SQUARE_SIZE * SQUARE_SIZE + SQUARE_SIZE * SQUARE_SIZE);
HALF_DIAGONAL = 0.5 * DIAGONAL
ANGLE_RANGE = radians(45.0)
ANGLE_PRECISION = radians(2.0)
# Golden Ratio
PHI = 0.5 * (-1.0 + sqrt(5.0))
# eighth of gesture length
START_ANGLE_INDEX = (NUM_POINTS / 8); 
ANGLE_SIMILARITY_THRESHOLD = radians(30.0)

class Rectangle():
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def __str__(self):
        return "Rectangle at ({},{}) of {} width and {} height".format(self.x, self.y, self.width, self.height)

    def __repr__(self):
        return "Rectangle at ({},{}) of {} width and {} height".format(self.x, self.y, self.width, self.height)

# Unistroke class: a unistroke template
class Unistroke():
    def __init__(self, name, useBoundedRotationInvariance, points):
        self.name = name
        self.points = resample(points, NUM_POINTS)
        radians = indicative_angle(self.points)
        self.points = rotate_by(self.points, -radians)
        # self.points = scale_dim_to(self.points, SQUARE_SIZE, ONE_D_THRESHOLD)
        if (useBoundedRotationInvariance):
            # restore
            self.points = rotate_by(self.points, radians)
        self.points = translate_to(self.points, ORIGIN)
        self.startUnitVector = calc_start_unit_vector(self.points, START_ANGLE_INDEX)
        # for Protractor
        self.vector = vectorize(self.points, useBoundedRotationInvariance)

    def __str__(self):
        return "\nUnistroke {}\nPoints {}".format(self.name, self.points)

    def __repr__(self):
        return "\nUnistroke {}\nPoints {}".format(self.name, self.points)

# Multistroke class: a container for unistrokes
#   [[(x,y),(x,y)],[(x,y),(x,y)]]
class Multistroke():
    def __init__(self, name, useBoundedRotationInvariance, strokes):
        self.name = name
        print(name)
        # number of individual strokes
        self.numStrokes = len(strokes)

        # array of integer indices
        order = [i for i in range(len(strokes))]
        # array of integer arrays
        orders = heap_permute(len(strokes), order, [])

        # returns array of point arrays
        unistrokes = make_unistrokes(strokes, orders)
        # unistrokes for this multistroke
        self.unistrokes = [Unistroke(name, useBoundedRotationInvariance, unistroke) for unistroke in unistrokes]

class Result():
    def __init__(self, name, score, ms):
        self.name = name;
        self.score = score;
        self.time = ms;


class Recognizer():
    def __init__(self, useBoundedRotationInvariance=False, requireSameNumStroke=False, useProtractor=False):
        # one predefined multistroke for each multistroke type
        self.useBoundedRotationInvariance = useBoundedRotationInvariance
        self.requireSameNumStroke = requireSameNumStroke
        self.useProtractor = useProtractor
        self.Multistrokes = [
            Multistroke("T", useBoundedRotationInvariance, 
            [
                [Point(30,7),Point(103,7)],
                [Point(66,7),Point(66,87)]
            ]),
            Multistroke("N", useBoundedRotationInvariance, 
            [
                [Point(177,92),Point(177,2)],
                [Point(182,1),Point(246,95)],
                [Point(247,87),Point(247,1)]
            ]),
            Multistroke("D", useBoundedRotationInvariance, 
            [
                [Point(345,9),Point(345,87)],
                [Point(351,8),Point(363,8),Point(372,9),Point(380,11),Point(386,14),Point(391,17),Point(394,22),Point(397,28),Point(399,34),Point(400,42),Point(400,50),Point(400,56),Point(399,61),Point(397,66),Point(394,70),Point(391,74),Point(386,78),Point(382,81),Point(377,83),Point(372,85),Point(367,87),Point(360,87),Point(355,88),Point(349,87)]
            ]),
            Multistroke("P", useBoundedRotationInvariance, 
            [
                [Point(507,8),Point(507,87)],
                [Point(513,7),Point(528,7),Point(537,8),Point(544,10),Point(550,12),Point(555,15),Point(558,18),Point(560,22),Point(561,27),Point(562,33),Point(561,37),Point(559,42),Point(556,45),Point(550,48),Point(544,51),Point(538,53),Point(532,54),Point(525,55),Point(519,55),Point(513,55),Point(510,55)]
            ]),
            Multistroke("X", useBoundedRotationInvariance, 
            [
                [Point(30,146),Point(106,222)],
                [Point(30,225),Point(106,146)]
            ]),
            Multistroke("H", useBoundedRotationInvariance, 
            [
                [Point(188,137),Point(188,225)],
                [Point(188,180),Point(241,180)],
                [Point(241,137),Point(241,225)]
            ]),
            Multistroke("I", useBoundedRotationInvariance, 
            [
                [Point(371,149),Point(371,221)],
                [Point(341,149),Point(401,149)],
                [Point(341,221),Point(401,221)]
            ]),
            Multistroke("exclamation", useBoundedRotationInvariance, 
            [
                [Point(526,142),Point(526,204)],
                [Point(526,221)]
            ]),
            Multistroke("line", useBoundedRotationInvariance, 
            [
                [Point(12,347),Point(119,347)]
            ]),
            Multistroke("five-point star", useBoundedRotationInvariance, 
            [
                [Point(177,396),Point(223,299),Point(262,396),Point(168,332),Point(278,332),Point(184,397)]
            ]),
            Multistroke("null", useBoundedRotationInvariance, 
            [
                [Point(382,310),Point(377,308),Point(373,307),Point(366,307),Point(360,310),Point(356,313),Point(353,316),Point(349,321),Point(347,326),Point(344,331),Point(342,337),Point(341,343),Point(341,350),Point(341,358),Point(342,362),Point(344,366),Point(347,370),Point(351,374),Point(356,379),Point(361,382),Point(368,385),Point(374,387),Point(381,387),Point(390,387),Point(397,385),Point(404,382),Point(408,378),Point(412,373),Point(416,367),Point(418,361),Point(419,353),Point(418,346),Point(417,341),Point(416,336),Point(413,331),Point(410,326),Point(404,320),Point(400,317),Point(393,313),Point(392,312)],
                [Point(418,309),Point(337,390)]
            ]),
            Multistroke("arrowhead", useBoundedRotationInvariance, 
            [
                [Point(506,349),Point(574,349)],
                [Point(525,306),Point(584,349),Point(525,388)]
            ]),
            Multistroke("pitchfork", useBoundedRotationInvariance, 
            [
                [Point(38,470),Point(36,476),Point(36,482),Point(37,489),Point(39,496),Point(42,500),Point(46,503),Point(50,507),Point(56,509),Point(63,509),Point(70,508),Point(75,506),Point(79,503),Point(82,499),Point(85,493),Point(87,487),Point(88,480),Point(88,474),Point(87,468)],
                [Point(62,464),Point(62,571)]
            ]),
            Multistroke("six-point star", useBoundedRotationInvariance, 
            [
                [Point(177,554),Point(223,476),Point(268,554),Point(183,554)],
                [Point(177,490),Point(223,568),Point(268,490),Point(183,490)]
            ]),
            Multistroke("asterisk", useBoundedRotationInvariance, 
            [
                [Point(325,499),Point(417,557)],
                [Point(417,499),Point(325,557)],
                [Point(371,486),Point(371,571)]
            ]),
            Multistroke("half-note", useBoundedRotationInvariance, 
            [
                [Point(546,465),Point(546,531)],
                [Point(540,530),Point(536,529),Point(533,528),Point(529,529),Point(524,530),Point(520,532),Point(515,535),Point(511,539),Point(508,545),Point(506,548),Point(506,554),Point(509,558),Point(512,561),Point(517,564),Point(521,564),Point(527,563),Point(531,560),Point(535,557),Point(538,553),Point(542,548),Point(544,544),Point(546,540),Point(546,536)]
            ])
        ]
        self.NumMultistrokes = len(self.Multistrokes)

    def recognize(self, strokes):
        if len(strokes) == 0:
            return Result("Null", 0.0, 0)
        t0 = time()
        # make one connected unistroke from the given strokes
        points = combine_strokes(strokes) 
        candidate = Unistroke("", self.useBoundedRotationInvariance, points)
        m = None
        b = inf
        n_best_list = {}
        # for each multistroke template
        for multistroke in self.Multistrokes:
            # optional -- only attempt match when same # of component strokes
            if not self.requireSameNumStroke or len(strokes) == multistroke.numStrokes:
                # for each unistroke within this multistroke
                for unistroke in multistroke.unistrokes:
                    # strokes start in the same direction
                    if angle_between_unit_vectors(candidate.startUnitVector, unistroke.startUnitVector) <= ANGLE_SIMILARITY_THRESHOLD:
                        d = None
                        if self.useProtractor:
                            # Protractor
                            d = optimal_cosine_distance(unistroke.vector, candidate.vector)
                        else:
                            # Golden Section Search (original $N)
                            d = distance_at_best_angle(candidate.points, unistroke, -ANGLE_RANGE, ANGLE_RANGE, ANGLE_PRECISION)
                        if (d < b):
                            # best (least) distance
                            b = d
                            # multistroke owner of unistroke
                            m = multistroke.name
                        n_best_list[multistroke.name] = (1.0 - d) if self.useProtractor else (1.0 - d / HALF_DIAGONAL)
        t1 = time()
        print(n_best_list)
        if n_best_list[m] >= 1:
            return Result("No match.", 0.0, t1-t0)
        else:
            return Result(m, (1.0 - b) if self.useProtractor else (1.0 - b / HALF_DIAGONAL), t1-t0)
    
    def add_gesture(self, name, strokes):
        multistroke = Multistroke(name, self.useBoundedRotationalInvariance, strokes)
        self.Multistrokes.append(multistroke)
    
    def delete_user_gestures(self):
        self.Multistrokes = self.Multistrokes[NUM_MULTISTROKES:]

#
# Private helper functions from here on down
#
def heap_permute(n, order, orders):
    newlist = []
    for p in permutations(order, n):
        newlist.append([i for i in p])
    return newlist
    
# def heap_permute(n, order, orders):
#     if (n == 1):
#         # append copy
#         orders.append(order.copy())
#         return orders
#     else:
#         for i in range(n):
#             orders = heap_permute(n - 1, order, orders)
#             if (n % 2 == 1):
#                 # swap 0, n-1
#                 order[0], order[n-1] = order[n-1], order[0]
#             else:
#                 # swap i, n-1
#                 order[i], order[n-1] = order[n-1], order[i]
#     return orders

def make_unistrokes(strokes, orders):
    # array of point arrays
    unistrokes = []
    for order in orders:
        # use b's bits for directions
        for b in range(pow(len(order), 2)):
            # array of points
            unistroke = []
            for i in range(len(order)):
                pts = []
                # is b's bit at index i on?
                if ((b >> i) & 1) == 1:
                    # copy and reverse
                    pts = strokes[order[i]].copy()[::-1]
                else:
                    # copy
                    pts = strokes[order[i]].copy()
                # append points
                for p in pts:
                    unistroke.append(p)
            # add one unistroke to set
            unistrokes.append(unistroke)
    return unistrokes

def combine_strokes(strokes):
    return [Point(p.x, p.y) for s in strokes for p in s]

def resample(points, n):
    # interval length
    I = path_length(points) / (n - 1)
    D = 0.0
    pts = points.copy()
    newpoints = [points[0]]
    i = 1
    while i < len(pts):
        d = distance(pts[i-1], pts[i])
        if ((D + d) >= I):
            q = Point(pts[i-1].x + ((I - D) / d) * (pts[i].x - pts[i-1].x), 
                      pts[i-1].y + ((I - D) / d) * (pts[i].y - pts[i-1].y))
            # append new point 'q'
            newpoints.append(q)
            # insert 'q' at position i in points s.t. 'q' will be the next i
            pts.insert(i, q) 
            D = 0.0
        else:
            D += d
        i += 1
    # somtimes we fall a rounding-error short of adding the last point, so add it if so
    if len(newpoints) == n - 1:
        newpoints.append(Point(points[-1].x, points[-1].y))
    return newpoints

def indicative_angle(points):
    c = centroid(points)
    return atan2(c.y - points[0].y, c.x - points[0].x)


# rotates points around centroid
def rotate_by(points, radians):
    c = centroid(points)
    cos_r = cos(radians)
    sin_r = sin(radians)
    newpoints = [Point((p.x - c.x) * cos_r - (p.y - c.y) * sin_r + c.x, 
                       (p.x - c.x) * sin_r + (p.y - c.y) * cos_r + c.y) for p in points]
    return newpoints


# scales bbox uniformly for 1D, non-uniformly for 2D
def scale_dim_to(points, size, ratio1D):
    B = bounding_box(points)
    newpoints = []
    point = B.width == 0.0 and B.height == 0.0 
    thin = B.width == 0.0 or B.height == 0.0
    for p in points:
        if point:
            newpoints.append(p)
            continue
        if thin:
            newpoints.append(Point(
                p.x * (size / max(B.height, B.width)),
                p.y * (size / max(B.height, B.width))
            ))
            continue
        # 1D or 2D gesture test
        if min(B.width / B.height, B.height / B.width) <= ratio1D:
            newpoints.append(Point(
                p.x * (size / max(B.height, B.width)),
                p.y * (size / max(B.height, B.width))
            ))
            continue
        else:
            newpoints.append(Point(
                p.x * (size / B.width),
                p.y * (size / B.height)
            ))
    return newpoints


# translates points' centroid
def translate_to(points, pt):
	c = centroid(points)
	newpoints = []
	for p in points:
		qx = p.x + pt.x - c.x
		qy = p.y + pt.y - c.y
		newpoints.append(Point(qx, qy))
	return newpoints


# for Protractor
def vectorize(points, useBoundedRotationInvariance):
    theta = atan2(points[0].y, points[0].x)
    delta = 0
    if useBoundedRotationInvariance:
        base_orientation = (PI / 4.0) *\
                            math.floor((theta + (PI / 8.0)))
        delta = base_orientation - theta
    else:
        delta = -1.0 * theta
    sum = 0
    vector = []
    for p in points:
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


# for Protractor
def optimal_cosine_distance(u, v):
    a = 0
    b = 0
    for i in range(0, len(u), 2):
        a = a + (u[i] * v[i]) + (u[i + 1] * v[i + 1])
        b = b + (u[i] * v[i + 1]) - (v[i] * u[i + 1])
    theta = math.atan(b / a)
    return math.acos(a * math.cos(theta) + b * math.sin(theta))


def distance_at_best_angle(points, T, a, b, threshold):
    x1 = PHI * a + (1.0 - PHI) * b
    f1 = distance_at_angle(points, T, x1)
    x2 = (1.0 - PHI) * a + PHI * b
    f2 = distance_at_angle(points, T, x2)
    while (fabs(b - a) > threshold):
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = PHI * a + (1.0 - PHI) * b
            f1 = distance_at_angle(points, T, x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = (1.0 - PHI) * a + PHI * b
            f2 = distance_at_angle(points, T, x2)
    return min(f1, f2)

# average distance between corresponding points in two paths
def path_distance(pts1, pts2):
    d = 0.0
    # assumes pts1.length == pts2.length
    for i, p in enumerate(pts1): 
        d += distance(p, pts2[i])
    return d / len(pts1)

def distance_at_angle(points, T, radians):
    newpoints = rotate_by(points, radians)
    return path_distance(newpoints, T.points)

def centroid(points):
    x, y = 0.0, 0.0
    for p in points:
        x += p.x
        y += p.y
    x /= len(points)
    y /= len(points)
    return Point(x, y)

def bounding_box(points):
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
def path_length(points):
    d = 0.0
    for i in range(1, len(points)):
        d += distance(points[i - 1], points[i])
    return d

# distance between two points
# def Distance(p1, p2):
#     return dist([p1.x, p1.y], [p2.x, p2.y])
def distance(p1, p2):
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    return sqrt(dx * dx + dy * dy)

# start angle from points[0] to points[index] normalized as a unit vector
def calc_start_unit_vector(points, index):
    index = int(index)
    v = Point(points[index].x - points[0].x, points[index].y - points[0].y)
    length = sqrt(v.x * v.x + v.y * v.y)
    return Point(v.x / length, v.y / length)

#  gives acute angle between unit vectors from (0,0) to v1, and (0,0) to v2
def angle_between_unit_vectors(v1, v2):
    n = v1.x * v2.x + v1.y * v2.y
    #  ensure [-1,+1]
    # arc cosine of the vector dot product
    return acos(max(-1.0, min(1.0, n)))