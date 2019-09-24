
import tkinter as tk
from model import *

win = tk.Tk()
canvas = tk.Canvas(win, bg='white', height=500, width=500)

class View:
	
	def __init__(self, model):
		win.title('view')
		win.geometry('800x600')		
		self.model = model
		self.draw()
		'''
		point = canvas.create_oval(300, 491, 305, 496, outline='pink', fill='pink')
		point = canvas.create_oval(100, 250, 105, 255, outline='pink', fill='pink')
		'''
		win.update()

	def draw(self):
		for x in range(0, 500, 50):
			line = canvas.create_line(-50, x, 550, x)
			line = canvas.create_line(x, -50, x, 550)
		for agent in self.model.agents:
			self.model.agents[agent].draw(canvas)
		canvas.pack()
		win.update()

if __name__ == '__main__':
	view = View(None)
	