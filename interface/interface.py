import Tkinter
from Tkinter import *
import networkx as nx
import matplotlib.pyplot as plt
import math
import random as rand
nodes = []
number_of_columns = 5 
number_of_rows = 5 
square_length = 50

selected_nodes = []
edge_set = {}
draw_mode = 0
fixed_tup = (0, 0)

class simpleapp_tk(Tkinter.Tk):
        global square_length, number_of_rows, number_of_columns
	def _create_circle(self, x, y, r, **kwargs):
		return self.create_oval(x-r, y-r, x+r, y+r, **kwargs)
	Tkinter.Canvas.create_circle = _create_circle
	def __init__(self, parent):
		Tkinter.Tk.__init__(self,parent)
		self.parent = parent
		self.square_length = square_length  
                self.color = StringVar()
                self.color.set("black")
                self.squares = []
		if draw_mode:
			self.draw_tree()
		else:
			self.initialize()
		
	def initialize(self):
		self.grid()
		self.labelVariable = Tkinter.StringVar()
                #initializing the set corresponding to each cell of grid
		self.initialize_nodes(number_of_rows * number_of_columns)
                #draw the canvas
                canvas = Tkinter.Canvas(height = self.square_length * number_of_rows , width = self.square_length * number_of_columns, bg = "blue")
		canvas.grid(column = 0, row = 5, sticky = 'EW')
                
                #draw the lines
                for i in range(number_of_rows):
                    for j in range(number_of_columns):
                        a = canvas.create_rectangle(j * self.square_length, i * self.square_length, (j + 1) * square_length, (i + 1) * square_length, fill = "black", outline = "red") 
                        self.squares.append(a)
		canvas.bind("<Key>", self.key)
		canvas.bind("<Button-1>", self.callback) 
                #draw the radio list
                R1 = Radiobutton(self, text = "gray", variable = self.color, anchor ="w", value = "gray")
                R1.grid(column = 5, row = 0, columnspan = 1, sticky = 'EW')

                R1 = Radiobutton(self, text = "blue", variable = self.color, anchor ="w", value = "blue")
                R1.grid(column = 5, row = 1, columnspan = 1, sticky = 'EW')

                R1 = Radiobutton(self, text = "white", variable = self.color, anchor ="w", value = "white")
                R1.grid(column = 5, row = 2, columnspan = 1, sticky = 'EW')
                

                R1 = Radiobutton(self, text = "yello", variable = self.color, anchor ="w", value = "yellow")
                R1.grid(column = 5, row = 3, columnspan = 1, sticky = 'EW')
	
		button = Tkinter.Button(self, text = u"finish", command = self.finish)
		button.grid(column = 1, row = 5)
	def onButtonClick(self):
		global number_of_columns
		global number_of_rows
		number_of_columns = self.w.get()
		number_of_rows = self.h.get()
		print(number_of_columns)
		print(number_of_rows)
        def key(self, event):
		print "pressed"
	def callback(self, event):
		canvas = event.widget
		x = canvas.canvasx(event.x)
		y = canvas.canvasy(event.y)
                index = (int)(x / self.square_length) + (int)(y / self.square_length) * number_of_columns
		nodes[index]  = self.color.get()
                print self.squares  
                #reconfiguring the square color
                canvas.itemconfig(self.squares[index], fill = self.color.get())

                print str(nodes) + "these are our nodes"
                print self.color.get() 
		canvas.create_circle(x, y, 3, fill="#BBB", outline="")
	def finish(self):
		self.destroy()	
	def initialize_nodes(self, num_of_clusters):
                #the colors in the all cells are 0
		for i in range(0, num_of_clusters):
			nodes.append(0)

if __name__ == "__main__":
	app = simpleapp_tk(None)
	app.title('app')
	app.mainloop()
	print "dd"



