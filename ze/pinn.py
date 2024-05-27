import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig



class motor():

	def __init__(self, system_params):
		self.system_params = system_params

		# System matrices
		self.A = [[-self.system_params["R"]/self.system_params["L"], -self.system_params["Ke"]/self.system_params["L"]], 
		     	  [self.system_params["Kt"]/self.system_params["J"], -self.system_params["b"]/self.system_params["J"]]]

		self.B = [[1/self.system_params["L"], 0],
		     	  [0, -1/self.system_params["J"]]]

		self.C = [[1, 0],
		     	  [0, 1]]

		self.D = [[0, 0],
		     	  [0, 0]]

		# Define system from matrices
		self.sys = sig.StateSpace(self.A, self.B, self.C, self.D)


	def run(self, t, inp):

		# Apply "u"" to the system
		t, y, _ = sig.lsim(self.sys, inp, t)

		return t,y
