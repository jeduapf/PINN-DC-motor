from ze import *
import plotly.graph_objects as go
import os
import torch
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def test_motor():
    # Simulation Parameters
    tstart = 0
    tstop = 5
    points = 5*10**3
    t = np.linspace(tstart, tstop+1, points)
    f = 13

    system_params = {   "J":0.1,
                        "b":0.5,
                        "Kt":1,
                        "L":1,
                        "R":5,
                        "Ke":1}
    inp = np.hstack((np.ones((len(t), 1)),0.006*np.sin(2*np.pi*f*t).reshape(-1,1)))
    
    moteur = motor(system_params)
    t,y = moteur.run(t, inp)

    fig = go.Figure()
    fig.add_trace(go.Scatter(   x=t, 
                                y=y[:,0], 
                                mode='lines', name = f"Current (A)",
                                line={
                                        "color":"rgba( 25, 0, 255, 0.8)",
                                    }))   

    fig.add_trace(go.Scatter(   x=t, 
                                y=y[:,1], 
                                mode='lines', name = f"Speed (rad/s)",
                                line={
                                        "color":"rgba( 255, 25, 0, 0.8)",
                                    }))  
    fig.update_layout(  height=700, width=1500, 
                        title_text=f"Ground truth current and speed", 
                        showlegend=True)
    fig.show()

if __name__ == "__main__":

    tstart = 0
    tstop = 5
    points = 5*10**3
    t = np.linspace(tstart, tstop+1, points)
    f = 13

    system_params = {
                    "t": t,
                    "J":0.1,
                    "b":0.5,
                    "Kt":1,
                    "L":1,
                    "R":5,
                    "Ke":1,
                    "motor_inputs": np.hstack((np.ones((len(t), 1)),0.006*np.sin(2*np.pi*f*t).reshape(-1,1)))} # [Vin, D]

    pinn_params = {     "DTYPE": torch.float32,
                        "physics_points": 500, 
                        "data_points": 100, 
                        "validation_points": 1000,
                        "neurons": 80,
                        "layers": 3,
                        "learning_rate": 10**-4,
                        # "regularization": {"lambda_I": 10**-2, "lambda_w": 10**-2,"lambda_f": 10**-2,"lambda_g": 10**-2},
                        "regularization": {"lambda_I": None, "lambda_w": None,"lambda_f": None,"lambda_g": None},
                        "epochs": 5*10**4,
                        "guesses": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]}

    # s = f'''b_{pinn_params['batch']}_n_{pinn_params['neurons']}_l_{pinn_params['layers']}_mu0_{pinn_params['mu_guess']:.1f}_k0_{pinn_params['k_guess']:.1f}_pys_{int(pinn_params['physics_points'])}_obs_{int(pinn_params['observation_points'])}_iter_{int(pinn_params['epochs']/1000)}k_lr_{pinn_params['learning_rate']:4.2e}_lb_{pinn_params['regularization_data']:4.2e}'''
    s = 'batata'
    SAVE_DIR, SAVE_GIF_DIR = DIRs(path_name = s, path_gif_name = 'gif')

    control_params = {  'SEARCH': False,
                        'PLOT': False,
                        'SAVE_DIR': SAVE_DIR,
                        'SAVE_GIF_DIR': SAVE_GIF_DIR,
                        'TEXT': True,
                        'SHOW_NET': False,
                        'SAVE_FILES' : False,
                        'FIGS': None}

    spring_pinn = PINN(control_params, system_params, pinn_params)
    spring_pinn.train()