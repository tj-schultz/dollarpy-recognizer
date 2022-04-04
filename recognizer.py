"""
name: recognizer.py -- dollarstore-recognizer
description: Recognizer class with member functions to resample, rotate, scale/translate paths and
run calculations to determine the score for a particular recognizer
authors: TJ Schultz, Skylar McCain
date: 1/27/22
"""
import math
import path as pth
import dollar



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