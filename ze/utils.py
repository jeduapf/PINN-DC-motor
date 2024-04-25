import scipy.io
import numpy as np
from PIL import Image
import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def save_paths():
    current_dir = os.getcwd()
    SAVE_DIR = os.path.join(current_dir,'Results')
    try:
        os.mkdir(SAVE_DIR)
    except:
        pass

    SAVE_DIR_GIF = os.path.join(SAVE_DIR,'gif_imgs')
    try:
        os.mkdir(SAVE_DIR_GIF)
    except:
        pass

    return SAVE_DIR, SAVE_DIR_GIF

def initialize_from_file(DATA_DIR):

    data = scipy.io.loadmat(DATA_DIR)

    time = data['time'].flatten()[:,None].squeeze()
    V_control = data['input'][:,0].flatten()[:,None].squeeze()
    Disturbance = data['input'][:,1].flatten()[:,None].squeeze()
    W_solution = data['output'][:,0].flatten()[:,None].squeeze()
    I_solution = data['output'][:,1].flatten()[:,None].squeeze()

    X_star = np.vstack((np.vstack((time,V_control)), Disturbance)).T
    u_star = np.vstack((W_solution, I_solution)).T

    # Domain bounds
    lb = time.min(0)
    ub = time.max(0)

    return X_star, u_star, lb, ub

def save_plotly(i, X_star, u_star, t_obs, u_obs, t_test, u_test, SAVE_DIR):

    fig = make_subplots(    rows=2, cols=1,
                            shared_xaxes=True, 
                            subplot_titles=("Speed (rad/s)","Current (A)"))
    fig.add_trace(
        go.Scatter( x=t_obs[:,0].detach().cpu().numpy(), 
                    y=u_obs[:,0].detach().cpu().numpy(), 
                    mode='markers', name = "Speed noisy observations",
                    marker={
                            "color":"rgba( 255, 25, 0, 0.55)",
                            'size': 10
                        }),
        row=1, col=1
        )
    fig.add_trace(
        go.Scatter( x=t_test[:,0].detach().cpu().numpy(), 
                    y=u_test[:,0].detach().cpu().numpy(), 
                    mode='lines', name = "PINN Speed solution",
                    line={
                            "color":"rgba( 0, 140, 210, 0.9)"
                        }),
        row=1, col=1
        )
    fig.add_trace(
        go.Scatter( x=X_star[:,0], 
                    y=u_star[:,0], 
                    mode='lines', name = "Speed Ground Truth",
                    line={
                            "color":"rgba( 25, 210, 0, 0.6)"
                        }),
        row=1, col=1
        )


    fig.add_trace(
        go.Scatter( x=t_obs[:,0].detach().cpu().numpy(), 
                    y=u_obs[:,1].detach().cpu().numpy(), 
                    mode='markers', name = "Current noisy observations",
                    marker={
                            "color":"rgba( 255, 25, 0, 0.55)",
                            'size': 10
                        }),
        row=2, col=1
        )
    fig.add_trace(
        go.Scatter( x=t_test[:,0].detach().cpu().numpy(), 
                    y=u_test[:,1].detach().cpu().numpy(), 
                    mode='lines', name = "PINN Current solution",
                    line={
                            "color":"rgba( 0, 140, 210, 0.9)"
                        }),
        row=2, col=1
        )
    fig.add_trace(
        go.Scatter( x=X_star[:,0], 
                    y=u_star[:,1], 
                    mode='lines', name = "Current Ground Truth",
                    line={
                            "color":"rgba( 25, 210, 0, 0.6)"
                        }),
        row=2, col=1
        )

    fig.update_layout(  height=1800, width=1200, 
                        title_text=f"PINN prediciton at iteration {i}", showlegend=False)
    fig.write_html(os.path.join(SAVE_DIR,f"prediciton_{i}.html"))

def save_gif_PIL(outfile, files, fps=5, loop=0):
    "Helper function for saving GIFs"
    imgs = [Image.open(file) for file in files]
    imgs[0].save(fp=outfile, format='GIF', append_images=imgs[1:], save_all=True, duration=int(1000/fps), loop=loop)

def generate_training_figures(inputs, track_constants, track_losses, SAVE_DIR):

    constants = np.array(track_constants)
    losses = np.array(track_losses)

    #
    # ************************************* LOSS *************************************
    #
    X = np.linspace(0, constants.shape[0] - 1, constants.shape[0] - 1, dtype= int) 
    fig_loss = make_subplots(   rows=5, cols=1, shared_xaxes=True, 
                                subplot_titles=( "Loss f", "Loss g", "Data Loss", "Total Loss", "Validation Loss"))

    fig_loss.add_trace(
        go.Scatter(x=X, y=losses[:,0], name = "Loss f"),
        row=1, col=1
        )

    fig_loss.add_trace(
        go.Scatter(x=X, y=losses[:,1], name = "Loss g"),
        row=2, col=1
        )

    fig_loss.add_trace(
        go.Scatter(x=X, y=losses[:,2], name = "Data Loss"),
        row=3, col=1
        )

    fig_loss.add_trace(
        go.Scatter(x=X, y=losses[:,3], name = "Total Loss"),
        row=4, col=1
        )

    fig_loss.add_trace(
        go.Scatter(x=X, y=losses[:,4], name = "Validation Loss"),
        row=5, col=1
        )
    fig_loss.update_xaxes(title_text="Iteration", row=5, col=1)

    fig_loss.update_layout( height=2000, width=1500, 
                            title_text="Losses training", showlegend=False)

    print(f"\n\n\t\t\tTotal Training Loss  = {np.sum(losses[:,3])}\n\t\t\tFinal Training Loss  = {losses[-1,3]}\n")
    print(f"\n\t\t\tTotal Validation Loss  = {np.sum(losses[:,4])}\n\t\t\tFinal Validation Loss  = {losses[-1,4]}\n\n")
    #
    # ************************************* CONSTANTS *************************************
    #
    fig_constants = make_subplots(   rows=6, cols=1, shared_xaxes=True, 
                                subplot_titles=( "J", "b", "Kt", "L", "R", "Ke"))

    fig_constants.add_trace(
        go.Scatter(x=X, y=constants[:,0]),
        row=1, col=1
        )
    fig_constants.add_hline(y=inputs["J"], line_dash="dot", row=1, col=1)
    fig_constants.update_xaxes(title_text="Iteration", row=1, col=1)

    fig_constants.add_trace(
        go.Scatter(x=X, y=constants[:,1]),
        row=2, col=1
        )
    fig_constants.add_hline(y=inputs["b"], line_dash="dot", row=2, col=1)
    fig_constants.update_xaxes(title_text="Iteration", row=2, col=1)

    fig_constants.add_trace(
        go.Scatter(x=X, y=constants[:,2]),
        row=3, col=1
        )
    fig_constants.add_hline(y=inputs["Kt"], line_dash="dot", row=3, col=1)
    fig_constants.update_xaxes(title_text="Iteration", row=3, col=1)

    fig_constants.add_trace(
        go.Scatter(x=X, y=constants[:,3]),
        row=4, col=1
        )
    fig_constants.add_hline(y=inputs["L"], line_dash="dot", row=4, col=1)
    fig_constants.update_xaxes(title_text="Iteration", row=4, col=1)

    fig_constants.add_trace(
        go.Scatter(x=X, y=constants[:,4]),
        row=5, col=1
        )
    fig_constants.add_hline(y=inputs["R"], line_dash="dot", row=5, col=1)
    fig_constants.update_xaxes(title_text="Iteration", row=5, col=1)

    fig_constants.add_trace(
        go.Scatter(x=X, y=constants[:,5]),
        row=6, col=1
        )
    fig_constants.add_hline(y=inputs["Ke"], line_dash="dot", row=6, col=1)
    fig_constants.update_xaxes(title_text="Iteration", row=6, col=1)

    fig_constants.update_layout( height=3000, width=1500, 
                            title_text="Losses training", showlegend=False)
    fig_constants.write_html(os.path.join(SAVE_DIR,"constants.html"))
    fig_loss.write_html(os.path.join(SAVE_DIR,"losses.html"))