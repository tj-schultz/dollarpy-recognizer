"""
name: main.py -- dollargeneral-recognizer
description: An implementation of the $n-recognizer in python with a Canvas input
authors: TJ Schultz, Skylar McCain
date: 4/4/22
"""
from datetime import datetime
import os.path
import time
import tkinter as tk
import canvas as cvs
import sys
import datetime
import xml.dom.minidom as xmlmd

import dollar
import path
from recognizer import Recognizer as uni_rec
from nrecognizer import Recognizer as multi_rec

info_text = "This is a tkinter application\n running on python %s and developed for open use by\n"\
    "TJ Schultz, Skylar McCain, Spencer Bass. 2022\n" \
    "\n\nDrag the mouse pointer to plot a single stroke path. Clicking again begins a new path.\n" \
    "After drawing, the recognizer will attempt to guess what you drew.\n"\
    "Click the checkbox to plot the path points." % (sys.version)


## prompts new info window for app
def info_window(text):
    window = tk.Toplevel()
    window.title("Info")

    ## open icon
    try:
        window.iconbitmap(os.path.join('resources', 'icon.ico'))
    except:
        print("Icon import error")

    #github_image = tk.PhotoImage(file=os.path.join('resources', 'qr.png'))

    info_label = tk.Label(window, text=text, justify="left")
    #github_label = tk.Label(window, image=github_image)
    info_label.pack(side="top", fill="both", expand=False)
    #github_label.pack(side="top", fill="both", expand=False)

## main application class defined for tkinter
class OnlineRecognition(tk.Frame):
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
        self.canvas.bind("<Button-3>", self.clear_canvas)
        self.canvas.bind("<Motion>", self.pathcanvas.draw_polyline)
        self.canvas.bind("<ButtonRelease-1>", self.update_path)

        ## pack canvas
        #self.canvas.grid(padx=100)
        self.canvas.pack(side="top", fill="both", expand=False)

        #GUI -- Clear Canvas and Recognize Button display
        self.clear_frame = tk.Frame(root)
        self.clear_button = tk.Button(self.clear_frame, width=15, text="Clear Canvas", command=self.clear_canvas)
        self.recog_button = tk.Button(self.clear_frame, width=20, text="Recognize", command=self.submit, bg="medium sea green")
        self.clear_frame.pack(side="top")
        self.clear_button.pack(side="left")
        self.recog_button.pack(side="right")


        ## GUI -- Recognizer display
        self.recog_frame = tk.Frame(root)
        self.match_label = tk.Label(self.recog_frame, text="Best Match:")
        self.match_entry = tk.Entry(self.recog_frame, width=20)
        self.recog_frame.pack(side="top")
        self.match_label.pack(side="left")
        self.match_entry.pack(side="left")


        ## GUI -- Path length display
        self.length_frame = tk.Frame(root)
        self.score_label = tk.Label(self.length_frame, text="Score:")
        self.score_entry = tk.Entry(self.length_frame, width=8)
        self.length_label = tk.Label(self.length_frame, text="Path Length:")
        self.length_entry = tk.Entry(self.length_frame, width=8)
        self.score_label.pack(side="left")
        self.score_entry.pack(side="left")
        self.length_frame.pack(side="top")
        self.length_label.pack(side="left")
        self.length_entry.pack(side="left")
        self.length_entry.insert(0, "0.0")


        ## GUI -- Path length -- Path Points display
        self.points_check = tk.Checkbutton(self.length_frame, text="Show points",\
                                           variable=self.show_points,\
                                           onvalue=True, offvalue=False)
        self.points_check.pack(side="left")

        ## GUI -- App info
        self.info_frame = tk.Frame(root)
        self.info_button = tk.Button(self.info_frame, text="?", command=lambda: info_window(info_text), font="d",\
                                     bg="medium sea green")
        self.info_button.pack(side="right")
        self.info_frame.pack(side="bottom")

        ## recognizer instantiation
        self.R = multi_rec()
        self.R_old = uni_rec()

    def clear_canvas(self, event=None):
        self.pathcanvas.clear()
        self.length_entry.delete(0, tk.END)
        self.length_entry.insert(0, 0.0)

    def submit(self):
        ## calculate results and update results entries
        npath = self.pathcanvas.npath
        nrecognizer = multi_rec(useBoundedRotationInvariance=True, useProtractor=True)
        print([s.parsed_path for s in npath.strokes])
        results = nrecognizer.recognize([s.parsed_path for s in npath.strokes])
        # if len(results) > 0:
        self.score_entry.delete(0, tk.END)
        self.score_entry.insert(0, round(results.score, 2))
        self.match_entry.delete(0, tk.END)
        self.match_entry.insert(0, results.name)

    ## returns pointer position
    def get_pointer_pos(self, event):
        print(event.x, event.y)
        return (event.x, event.y)

    ## calls pen to end drawing and updates the path length entry, then calls method to recognize
    def update_path(self, event):

        ## stops pen
        self.pathcanvas.pen(event)


        ## updates previous path length display
        length = float(self.length_entry.get())
        self.length_entry.delete(0, tk.END)
        self.length_entry.insert(0, round(self.R_old.path_length(self.pathcanvas.path), 2) + length)
        self.length_entry.update()

        ## resample path
        self.pathcanvas.resampled = self.R_old.resample(self.pathcanvas.path, dollar.Dollar.prefs["n_points"])

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


class DataCollection(tk.Frame):
    sample_types = ["T", "N", "D", "P", "X", "H", "I", "exclamation", "line",\
                    "five-point star", "null", "arrowhead", "pitchfork", "six-point star",\
                    "asterisk", "half-note"]
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

        
        ## create a subjectID input above canvas
        self.subject_frame = tk.Frame(root)
        self.subject_label = tk.Label(self.subject_frame, text="SubjectID:")
        self.subject_entry = tk.Entry(self.subject_frame, width=20, text="NULL")
        self.subject_frame.pack(side="top")
        self.subject_entry.pack(side="right")
       
        ## create PathCanvas container object for Canvas
        self.pathcanvas = cvs.PathCanvas(root, self.canvas, C_WIDTH, C_HEIGHT)


        ## binding mouse events to Canvas object in tk frame, and tying them to functions in PathCanvas object
        self.canvas.bind("<Button-1>", self.pathcanvas.pen)
        self.canvas.bind("<Button-3>", self.clear_canvas)
        self.canvas.bind("<Motion>", self.pathcanvas.draw_polyline)
        self.canvas.bind("<ButtonRelease-1>", self.update_path)

        ## pack canvas
        #self.canvas.grid(padx=100)
        self.canvas.pack(side="top", fill="both", expand=False)
        
        #GUI -- Clear Canvas and Log Gesture
        self.clear_frame = tk.Frame(root)
        self.clear_button = tk.Button(self.clear_frame, width=15, text="Clear Canvas", command=self.clear_canvas)
        self.submit_button = tk.Button(self.clear_frame, width=20, text="Log Gesture", command=self.submit)
        self.clear_frame.pack(side="top")
        self.clear_button.pack(side="left")
        self.submit_button.pack(side="right")
        self.subject_label.pack(side="bottom")

        ## GUI -- Prompt display
        self.prompt_frame = tk.Frame(root)
        self.drawing_label = tk.Label(self.prompt_frame, text="Drawing:")
        self.samplenum_label = tk.Label(self.prompt_frame, text="Sample#:")
        self.drawing_entry = tk.Entry(self.prompt_frame, width=24)
        self.samplenum_entry = tk.Entry(self.prompt_frame, width=3)
        
        self.prompt_frame.pack(side="top")
        self.drawing_label.pack(side="left")
        self.drawing_entry.pack(side="left")
        self.samplenum_entry.pack(side="right")
        self.samplenum_label.pack(side="right")

        self.drawing_entry.insert(0, self.sample_types[0])
        self.samplenum_entry.insert(0, 1)
        
        # ## GUI -- Recognizer display
        # self.recog_frame = tk.Frame(root)
        # self.match_label = tk.Label(self.recog_frame, text="Best Match:")
        # self.match_entry = tk.Entry(self.recog_frame, width=32)
        # self.recog_frame.pack(side="top")
        # self.match_label.pack(side="left")
        # self.match_entry.pack(side="left")


        ## GUI -- Path length and submission display
        self.length_frame = tk.Frame(root)
        self.length_label = tk.Label(self.length_frame, text="Path Length:")
        self.length_entry = tk.Entry(self.length_frame, width=8)
        self.submit_button = tk.Button(self.length_frame, width=16, text="Log Sample", command=self.submit)
        self.length_frame.pack(side="top")
        self.length_label.pack(side="left")
        self.length_entry.pack(side="left")
        self.length_entry.insert(0, "0.0")



        ## GUI -- Path length -- Path Points display
        self.points_check = tk.Checkbutton(self.length_frame, text="Show points",\
                                           variable=self.show_points,\
                                           onvalue=True, offvalue=False)
        self.points_check.pack(side="left")

        ## GUI -- App info
        self.info_frame = tk.Frame(root)
        self.info_button = tk.Button(self.info_frame, text="?", command=lambda: info_window(info_text), font="d",\
                                     bg="medium sea green")
        self.info_button.pack(side="right")
        self.info_frame.pack(side="bottom")

        ## recognizer instantiation
        self.R = multi_rec()
        self.R_old = uni_rec()


    ## submits path to file
    def submit(self):
        ## get current loop information
        s_index = 0
        s_id = self.subject_entry.get()
        if s_id == "":
            info_window("Error: Please enter a subject ID.")
            return
        g_id = int(self.samplenum_entry.get())
        name = self.drawing_entry.get()
        s_index = self.sample_types.index(name)
        name += str(g_id).zfill(2)

        ## write out
        to_xml(path=self.pathcanvas.npath, name=name, s_id=s_id, g_id=g_id)

        ## increment loop
        self.samplenum_entry.delete(0, tk.END)
        if g_id == 10:
            self.samplenum_entry.insert(0, 1)
            self.drawing_entry.delete(0, tk.END)
            if s_index < (len(self.sample_types) - 1):
                s_index = s_index + 1
            else:
                s_index = 0
                self.subject_entry.delete(0, tk.END)
                self.subject_entry.insert(0, "NULL")
            self.drawing_entry.insert(0, self.sample_types[s_index])
        else:
            self.samplenum_entry.insert(0, g_id + 1)

        ## updates previous path length display
        length = float(self.length_entry.get())
        self.length_entry.delete(0, tk.END)
        self.length_entry.insert(0, round(self.R_old.path_length(self.pathcanvas.path), 2) + length)
        self.length_entry.update()

        self.pathcanvas.clear()
        self.length_entry.delete(0, tk.END)
        self.length_entry.insert(0, 0.0)

        self.clear_canvas()

    ## returns pointer position
    def get_pointer_pos(self, event):
        print(event.x, event.y)
        return (event.x, event.y)

    ## calls pen to end drawing and updates the path length entry, then calls method to recognize
    def update_path(self, event):

        ## stops pen
        self.pathcanvas.pen(event)

        ## resample path
        self.pathcanvas.resampled = self.R_old.resample(self.pathcanvas.path, dollar.Dollar.prefs["n_points"])

        ## draws points if show_points
        if self.show_points.get():
            self.pathcanvas.draw_points(self.pathcanvas.resampled, cvs.line_pref["point_fill"])

    def clear_canvas(self, event=None):
        self.pathcanvas.clear()
        self.length_entry.delete(0, tk.END)
        self.length_entry.insert(0, 0.0)
    
    ## xml path-to-file method

def to_xml(path, name, s_id, g_id, speed="medium"):

    ## create doc object
    doc = xmlmd.Document()

    ## create root
    root = doc.createElement("Gesture")

    ## write all root attributes
    root.setAttribute("Name", name)
    root.setAttribute("Subject", s_id)
    root.setAttribute("Speed", speed)
    root.setAttribute("Number", str(g_id))
    root.setAttribute("NumPts", str(len(path)))
    root.setAttribute("AppName", "dollarstore-notepad")
    root.setAttribute("Date", str(datetime.datetime.now().date()))
    root.setAttribute("Time", path.strokes[0].times[0])

    ## append root element
    doc.appendChild(root)
    for s in path.strokes:
        stroke = doc.createElement("Stroke")
        i = 0
        for p in s.parsed_path:
            point = doc.createElement("Point")
            point.setAttribute("X", str(p.x))
            point.setAttribute("Y", str(p.y))
            point.setAttribute("T", s.times[i])
            stroke.appendChild(point)
            i+=1
        root.appendChild(stroke)

    ## write file out
    try:
        os.mkdir(s_id)
    except:
        print("")
    with open("%s/%s.xml" % (s_id, ("%s-%s" % (s_id, name))), 'w') as f:
        doc.writexml(f,
                     indent="  ",
                     addindent="  ",
                     newl='\n')


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


class SelectionWindow(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        self.clear_frame = tk.Frame(root)
        
        self.online_button = tk.Button(self.clear_frame, width=20, text="Online Recognition", command=self.online)
        
        self.data_collection_button = tk.Button(self.clear_frame, width=20, text="Data Collection", command=self.data_collection)
        
        self.clear_frame.pack(side="top")
        self.online_button.pack(side="left")
        self.data_collection_button.pack(side="right")

    def online(self):
        self.online_button.pack_forget()
        self.data_collection_button.pack_forget()
        window = tk.Toplevel()
        OnlineRecognition(window).pack(side="top", fill="both", expand=True)


    def data_collection(self):
        self.online_button.pack_forget()
        self.data_collection_button.pack_forget()
        window = tk.Toplevel()
        DataCollection(window).pack(side="top", fill="both", expand=True)


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
    SelectionWindow(root).pack(side="top", fill="both", expand=True)

    ## run loop
    root.mainloop()