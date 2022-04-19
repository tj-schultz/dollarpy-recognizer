import unittest
from math import radians, degrees
from nrecognizer import Point
from nrecognizer import heap_permute, combine_strokes, centroid, rotate_by, indicative_angle, translate_to, calc_start_unit_vector, vectorize, angle_between_unit_vectors, optimal_cosine_distance

class TestRecognizer(unittest.TestCase):

    def test_heap_permute_single(self):
        single = [0]
        orders = []
        output = heap_permute(1, single, orders)
        self.assertEqual(output, [[0]])

    def test_heap_permute_double(self):
        double = [0, 1]
        orders = []
        output = heap_permute(2, double, orders)
        self.assertEqual(output, [[0,1], [1,0]])

    def test_heap_permute_triple(self):
        triple = [0, 1, 2]
        orders = []
        output = heap_permute(3, triple, orders)
        self.assertEqual(output, [[0,1,2], [0,2,1], [1,0,2], [1,2,0], [2,0,1], [2,1,0]])

    def test_combine_strokes(self):
        strokes = [[Point(0,0), Point(0,1)], [Point(0,1), Point(1,0)]]
        output = combine_strokes(strokes)
        self.assertEqual(str(output), str([Point(0,0), Point(0,1), Point(0,1), Point(1,0)]))
    
    def test_centroid(self):
        points = [Point(0,0)]
        output = centroid(points)
        # Centroid of (0,0) is itself
        self.assertEqual(output, Point(0,0))
        
        points.append(Point(0,1))
        output = centroid(points)
        # Centroid of (0,0), (0,1) is (0, 0.5)
        self.assertEqual(output, Point(0, 0.5))
        
        points.append(Point(1,1))
        points.append(Point(1,0))
        output = centroid(points)
        # Centroid of (0,0),(0,1),(1,0),(1,1) is (0.5, 0.5)
        self.assertEqual(output, Point(0.5, 0.5))

    def test_rotate(self):
        points = [Point(0,0)]
        output = rotate_by(points, radians(90))
        self.assertEqual(output, [Point(0,0)])
        
        points.append(Point(0,1))
        output = rotate_by(points, radians(90))
        self.assertEqual(str(output), str([Point(0.5, 0.5), Point(-0.5, 0.5)]))

        points.append(Point(1,1))
        points.append(Point(1,0))
        output = rotate_by(points, radians(90))
        self.assertEqual(str(output), str([Point(1,0), Point(0,0), Point(0,1), Point(1,1)]))

    def test_indicative_angle(self):
        points = [Point(0,0)]
        output = indicative_angle(points)
        self.assertEqual(degrees(output), 0.0)
        
        points.append(Point(0,1))
        output = indicative_angle(points)
        self.assertEqual(degrees(output), 90.0)

        points.append(Point(1,1))
        points.append(Point(1,0))
        output = indicative_angle(points)
        self.assertEqual(degrees(output), 45.0)

    def test_translate_to(self):
        points = [Point(0,0)]
        origin = Point(0,0)
        output = translate_to(points, origin)
        self.assertEqual(output, [origin])
        
        points.append(Point(0,1))
        output = translate_to(points, origin)
        self.assertEqual(output, [Point(0,-0.5),Point(0,0.5)])

        points.append(Point(1,1))
        points.append(Point(1,0))
        output = translate_to(points, origin)
        self.assertEqual(output, [Point(-0.5,-0.5),Point(-0.5,0.5),Point(0.5,0.5),Point(0.5,-0.5)])

    def test_vectorize(self):
        points = [Point(0,0)]
        output = vectorize(points, False)

        points.append(Point(1,2))
        output = vectorize(points, False)
        print(output)



if __name__ == '__main__':
    unittest.main()