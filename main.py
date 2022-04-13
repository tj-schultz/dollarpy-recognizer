"""
name: main.py -- dollargeneral-recognizer
description: An implementation of the $n-recognizer in python with a Canvas input
authors: TJ Schultz, Skylar McCain
date: 4/4/22
"""
import os.path
import time
import tkinter as tk
import canvas as cvs
import sys

import dollar
import path
import recognizer as rec

## main application class defined for tkinter
class MainApplication(tk.Frame):
    def __init__(self, parent, *args, **kwargs):

        tk.Frame.__init__(self, parent, *args, **kwargs)

        self.parent = parent

        ## variable to show points when drawing
        self.show_points = tk.BooleanVar()

        ## GUI -- Canvas

        ## define Canvas properties
        C_WIDTH = 500
        C_HEIGHT = 500

        self.canvas = tk.Canvas(root, width=C_WIDTH, height=C_HEIGHT, bg="lightgrey", \
                                         highlightthickness=5, highlightbackground="medium sea green")

        ## create PathCanvas container object for Canvas
        self.pathcanvas = cvs.PathCanvas(root, self.canvas, C_WIDTH, C_HEIGHT)


        ## binding mouse events to Canvas object in tk frame, and tying them to functions in PathCanvas object
        self.canvas.bind("<Button-1>", self.pathcanvas.pen)
        self.canvas.bind("<Motion>", self.pathcanvas.draw_polyline)
        self.canvas.bind("<ButtonRelease-1>", self.update_path)

        ## pack canvas
        self.canvas.grid(padx=100)
        self.canvas.pack(side="top", fill="both", expand=False)


        ## GUI -- Recognizer display
        self.recog_frame = tk.Frame(root)
        self.score_label = tk.Label(self.recog_frame, text="Score:")
        self.score_entry = tk.Entry(self.recog_frame, width=8)
        self.match_label = tk.Label(self.recog_frame, text="Best Match:")
        self.match_entry = tk.Entry(self.recog_frame, width=32)
        self.recog_button = tk.Button(self.recog_frame, width=64, text="Recognize Gesture", command=self.submit([]))
        self.recog_frame.pack(side="top")
        self.match_label.pack(side="left")
        self.match_entry.pack(side="left")
        self.score_label.pack(side="left")
        self.score_entry.pack(side="left")
        self.recog_button.pack(side="top")

        ## GUI -- Path length display
        self.length_frame = tk.Frame(root)
        self.length_label = tk.Label(self.length_frame, text="Path Length:")
        self.length_entry = tk.Entry(self.length_frame, width=8)
        self.length_frame.pack(side="top")
        self.length_label.pack(side="left")
        self.length_entry.pack(side="left")



        ## GUI -- Path length -- Path Points display
        self.points_check = tk.Checkbutton(self.length_frame, text="Show points",\
                                           variable=self.show_points,\
                                           onvalue=True, offvalue=False)
        self.points_check.pack(side="left")

        ## GUI -- App info
        self.info_frame = tk.Frame(root)
        self.info_button = tk.Button(self.info_frame, text="?", command=self.info_window, font="d",\
                                     bg="medium sea green")
        self.info_button.pack(side="right")
        self.info_frame.pack(side="bottom")

        ## recognizer instantiation
        self.R = rec.NDollarRecognizer()

    ## prompts new info window for app
    def info_window(self):
        window = tk.Toplevel()
        window.title("Info")

        ## open icon
        try:
            window.iconbitmap(os.path.join('resources', 'icon.ico'))
        except:
            print("Icon import error")

        info_text = "This is a tkinter application\n running on python %s and developed for open use by\n"\
        "TJ Schultz, Skylar McCain. 2022\n" \
        "\n\nDrag the mouse pointer to plot a single stroke path. Clicking again begins a new path.\n" \
        "After drawing, the recognizer will attempt to guess what you drew.\n"\
                "Click the checkbox to plot the path points." % (sys.version)

        #github_image = tk.PhotoImage(file=os.path.join('resources', 'qr.png'))

        info_label = tk.Label(window, text=info_text, justify="left")
        #github_label = tk.Label(window, image=github_image)
        info_label.pack(side="top", fill="both", expand=False)
        #github_label.pack(side="top", fill="both", expand=False)

    def submit(self, npath):
        ## calculate results and update results entries
        nrecognizer = rec.NDollarRecognizer()
        results = nrecognizer.Recognize(npath, preprocess=True)
        results = []
        if len(results) > 0:
            self.score_entry.delete(0, tk.END)
            self.score_entry.insert(0, round(results[0][1], 2))
            self.match_entry.delete(0, tk.END)
            self.match_entry.insert(0, results[0][0])

    ## returns pointer position
    def get_pointer_pos(self, event):
        print(event.x, event.y)
        return (event.x, event.y)

    ## calls pen to end drawing and updates the path length entry, then calls method to recognize
    def update_path(self, event):

        ## stops pen
        self.pathcanvas.pen(event)

        ## updates previous path length display
        self.length_entry.delete(0, tk.END)
        self.length_entry.insert(0, round(self.R.path_length(self.pathcanvas.path), 2))
        self.length_entry.update()

        ## resample path
        self.pathcanvas.resampled = self.R.resample(self.pathcanvas.path, dollar.Dollar.prefs["n_points"])

        ## draws points if show_points
        if self.show_points.get():
            self.pathcanvas.draw_points(self.pathcanvas.resampled, cvs.line_pref["point_fill"])



        """
        test = self.R.preprocess(self.pathcanvas.path)
        test_temp = self.R.preprocess(dollar.Dollar.templates["zig-zag"])
        for p in test.parsed_path:
            p.x += 150
            p.y += 150
        for q in test_temp.parsed_path:
            q.x += 150
            q.y += 150
        

        self.pathcanvas.draw_points(test, color="blue")
        self.pathcanvas.draw_points(test_temp, color="green")
        """


if __name__ == "__main__":

    ## tkinter application root
    root = tk.Tk()

    try:
        ## open icon
        root.iconbitmap(os.path.join('resources', 'icon.ico'))
    except:
        print("Icon import error")



    ## define window properties
    root.title("dollargeneral-recognizer")
    root.minsize(500, 700)
    root.maxsize(500, 700)

    ## organize root geometry as window
    MainApplication(root).pack(side="top", fill="both", expand=True)

    ## run loop
    root.mainloop()