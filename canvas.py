"""
name: canvas.py -- dollargeneral-recognizer
description: Canvas tkinter definitions
authors: TJ Schultz, Skylar McCain, Spencer Bass
date: 4/20/22
"""

import tkinter as tk
import path as pth
import time

## line attributes
line_pref = {
    "width": 5,
    "fill": "black",
    "point_fill": "medium sea green",
    "recog_fill": "chocolate1",
    "ssteps": 5
}

## PathCanvas class
class PathCanvas():

    ## time constant to wait before parsing a new x, y value from mouse motion
    PATH_PARSE_CONST = 0.001

    ## class members
    down = False
    plotting = False
    path = pth.Path()
    npath = pth.N_Path()
    resampled = pth.Path()

    C_WIDTH = 0
    C_HEIGHT = 0

    def __init__(self, parent, _canvas, WIDTH, HEIGHT, down=False):

        self.parent = parent
        self.down = down

        ## create native tkinter canvas object inside PathCanvas
        ## set canvas to _canvas
        self.canvas = _canvas

        ## pack
        self.canvas.pack(side="top", fill="none", expand=True)

        ## migrate passed height, width consts
        self.C_WIDTH = WIDTH
        self.C_HEIGHT = HEIGHT

    ## returns pointer position
    def get_pointer_pos(self, event):
        print(event.x, event.y)
        return (event.x, event.y)

    ## toggle boolean state for recording pen values
    def toggle_down(self):
        self.down = not self.down

    ## pen tool method to draw path
    def pen(self, event):
        if not self.down:

            self.toggle_down()
            start_x = event.x
            start_y = event.y
            self.path = pth.Path(pth.Point(start_x, start_y))
        else:
            self.npath.stitch(self.path)
            self.toggle_down()

    ## log motion to the path structure using stitch
    def log_motion(self, x, y):
        time.sleep(self.PATH_PARSE_CONST)
        if self.down:
            self.path.stitch(pth.Point(x, y))
            #print(str(self.path))

    ## draws polyline on canvas for last two points
    def draw_polyline(self, event):
        if not self.plotting:
            self.plotting = True
            ## simply plot the line in one pass
            if event == None:
                for i in range(len(self.path)-1):
                    self.canvas.create_line(self.path.parsed_path[-(i + 2)].x, self.path.parsed_path[-(i + 2)].y, \
                                            self.path.parsed_path[-(i + 1)].x, self.path.parsed_path[-(i + 1)].y, \
                                            width=line_pref["width"], fill=line_pref["fill"], \
                                            splinesteps=line_pref["ssteps"], capstyle="round")
            ## prevents drawing beyond border during active drawing
            elif event.x <= self.C_WIDTH and event.y <= self.C_HEIGHT:

                ## log motion
                self.log_motion(event.x, event.y)

                ## if 2 or more points in path, draw a line with the last two points
                if self.path and len(self.path) > 1 and self.down:
                    self.canvas.create_line(self.path.parsed_path[-2].x, self.path.parsed_path[-2].y, \
                                            self.path.parsed_path[-1].x, self.path.parsed_path[-1].y,\
                                            width=line_pref["width"], fill=line_pref["fill"],\
                                            splinesteps=line_pref["ssteps"], capstyle="round")
                ## update the Canvas object
                self.canvas.update()
            self.plotting = False

    ## draws points on canvas for completed path
    def draw_points(self, path, color="black"):

        ## if 2 or more points in path, draw the points with a 'shown' color or 'hidden' color
        if len(path) > 1:
            for p in path.parsed_path:
                self.canvas.create_oval(p.x - 2, p.y - 2, \
                                        p.x + 2, p.y + 2, \
                                        fill=color)
        ## update the Canvas object
        self.canvas.update()

    ## clears the canvas
    def clear(self):
        ## dealloc list of Point objects
        self.npath = pth.N_Path()

        ## clear Canvas object of all drawings
        self.canvas.delete("all")