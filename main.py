from ze import *
import plotly.graph_objects as go
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

if __name__ == "__main__":

	# Simulation Parameters
	tstart = 0
	tstop = 5
	points = 5*10**3
	t = np.linspace(tstart, tstop+1, points)
	f = 13

	system_params = {	"J":0.1,
						"b":0.5,
						"Kt":1,
						"L":1,
						"R":5,
						"Ke":1}
	inp = np.hstack((np.ones((len(t), 1)),0.006*np.sin(2*np.pi*f*t).reshape(-1,1)))
	
	moteur = motor(system_params)
	t,y = moteur.run(t, inp)

	fig = go.Figure()
	fig.add_trace(go.Scatter( 	x=t, 
			                    y=y[:,0], 
			                    mode='lines', name = f"Current (A)",
			                    line={
			                            "color":"rgba( 25, 0, 255, 0.8)",
			                        }))   

	fig.add_trace(go.Scatter( 	x=t, 
			                    y=y[:,1], 
			                    mode='lines', name = f"Speed (rad/s)",
			                    line={
			                            "color":"rgba( 255, 25, 0, 0.8)",
			                        }))  
	fig.update_layout(  height=700, width=1500, 
                        title_text=f"Ground truth current and speed", 
                        showlegend=True)
	fig.show()
