import scipy.io
import numpy as np
from PIL import Image
import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

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

def save_plt(i, t_obs_np, u_obs_np, t_test_np, u_test_np, u_test_star_np, SAVE_DIR_GIF):

    residual = np.abs(u_test_star_np - u_test_np)
    test_p = residual.shape[0]
    sum_res = np.sum(residual, axis = 0)

    fig, ax = plt.subplots(3,1)
    fig.set_size_inches(18.5, 10.5)
    fig.suptitle(f"Observations and PINN approximation at iter: {i}")

    plt.subplot(3,1,1)
    plt.scatter(t_obs_np, u_obs_np[:,0], label="Speed noisy observations", alpha=0.6)
    plt.plot(t_test_np, u_test_np[:,0], label="PINN solution", color="tab:green")
    plt.legend()

    plt.subplot(3,1,2)
    plt.scatter(t_obs_np, u_obs_np[:,1], label="Current noisy observations", alpha=0.6)
    plt.plot(t_test_np, u_test_np[:,1], label="PINN solution", color="tab:green")
    plt.legend()

    plt.subplot(3,1,3)
    plt.plot(t_test_np, residual[:,0], label=f"Speed Residual - {test_p} Points (Sum = {sum_res[0]})",color="red", alpha=0.7)
    plt.plot(t_test_np, residual[:,1], label=f"Current Residual - {test_p} Points (Sum = {sum_res[1]})", color="black", alpha=0.7)
    plt.legend()

    return fig

def save_plotly(i, X_star, u_star, t_obs_np, u_obs_np, u_test_star_np, t_test_np, u_test_np, SAVE_DIR):
    res = np.abs(u_test_star_np-u_test_np)
    speed_res = res[:,0]
    curr_res = res[:,1]

    fig = make_subplots(    rows=4, cols=1,
                            shared_xaxes=True, 
                            subplot_titles=(
                            "Speed (rad/s)", f"Speed Residual Test Points - {len(speed_res)} (Sum = {np.sum(speed_res):.3f})", 
                            "Current (A)", f"Current Residual Test Points - {len(speed_res)} (Sum = {np.sum(curr_res):.3f})"))
    fig.add_trace(
        go.Scatter( x=t_obs_np, 
                    y=u_obs_np[:,0], 
                    mode='markers', name = "Speed noisy observations",
                    marker={
                            "color":"rgba( 255, 25, 0, 0.55)",
                            'size': 10
                        }),
        row=1, col=1
        )   
    fig.add_trace(
        go.Scatter( x=t_test_np, 
                    y=u_test_np[:,0], 
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
        go.Scatter( x=t_test_np, 
                    y=speed_res, 
                    mode='lines', name = f"Speed Residual (Sum = {np.sum(speed_res):.3f})",
                    line={
                            "color":"rgba( 255, 20, 0, 0.8)"
                        }),
        row=2, col=1
        )


    fig.add_trace(
        go.Scatter( x=t_obs_np, 
                    y=u_obs_np[:,1], 
                    mode='markers', name = "Current noisy observations",
                    marker={
                            "color":"rgba( 255, 25, 0, 0.55)",
                            'size': 10
                        }),
        row=3, col=1
        )
    fig.add_trace(
        go.Scatter( x=t_test_np, 
                    y=u_test_np[:,1], 
                    mode='lines', name = "PINN Current solution",
                    line={
                            "color":"rgba( 0, 140, 210, 0.9)"
                        }),
        row=3, col=1
        )
    fig.add_trace(
        go.Scatter( x=X_star[:,0], 
                    y=u_star[:,1], 
                    mode='lines', name = "Current Ground Truth",
                    line={
                            "color":"rgba( 25, 210, 0, 0.6)"
                        }),
        row=3, col=1
        )

    fig.add_trace(
        go.Scatter( x=t_test_np, 
                    y=curr_res, 
                    mode='lines', name = f"Current Residual (Sum = {np.sum(curr_res):.3f})",
                    line={
                            "color":"rgba( 255, 20, 0, 0.8)"
                        }),
        row=4, col=1
        )

    fig.update_layout(  height=1800, width=1500, 
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
                            title_text="Constants training", showlegend=False)
    fig_constants.write_html(os.path.join(SAVE_DIR,"constants.html"))
    fig_loss.write_html(os.path.join(SAVE_DIR,"losses.html"))

def deep_dive_plot( u_obs_hat_np, u_obs_np, t_physics_np, V_in_np, W_pert_np,w_phy_hat_np, i_phy_hat_np, dwdt_phy_hat_np, didt_phy_hat_np, d2wdt2_phy_hat_np,J_np, b_np, Kt_np, L_np, R_np, Ke_np, SAVE_DIR):
  
    fig = make_subplots(rows=5, cols=2,
                        subplot_titles=(
                        r"$f: \frac{\partial^2 \omega}{\partial t^2}$", r"$g: \frac{\partial i}{\partial t}$",
                        r"$f: \frac{\partial \omega}{\partial t}$", r"$g: \frac{\partial \omega}{\partial t}$",
                        r"$f: i$", r"$g: i$",
                        r"$f: W_{perturbation}$", r"$g: V_{input}$",
                        r"$f: \lvert J\frac{\partial^2 \omega}{\partial t^2} + b\frac{\partial \omega}{\partial t} - K_{t}i + W_{perturbation} \lvert^2$", r"$g: \lvert L\frac{\partial i}{\partial t} + K_{e}\frac{\partial \omega}{\partial t} + Ri - V_{input} \lvert^2$"))
    # ------------------------------------------ f physics ------------------------------------------
    fig.add_trace(
        go.Scatter( x=t_physics_np, 
                    y=d2wdt2_phy_hat_np, 
                    mode='lines', name = f"J = {J_np:.3f}",
                    line={
                            "color":"rgba( 255, 25, 0, 0.8)",
                        }),
        row=1, col=1
        )   
    fig.add_trace(
        go.Scatter( x=t_physics_np, 
                    y=dwdt_phy_hat_np, 
                    mode='lines', name = f"b = {b_np:.3f}",
                    line={
                            "color":"rgba( 255, 25, 0, 0.8)",
                        }),
        row=2, col=1
        )  
    fig.add_trace(
        go.Scatter( x=t_physics_np, 
                    y=i_phy_hat_np, 
                    mode='lines', name = f"Kt = {Kt_np:.3f}",
                    line={
                            "color":"rgba( 255, 25, 0, 0.8)",
                        }),
        row=3, col=1
        ) 
    fig.add_trace(
        go.Scatter( x=t_physics_np, 
                    y=W_pert_np, 
                    mode='lines', name = r"$W_{perturbation}$",
                    line={
                            "color":"rgba( 255, 25, 0, 0.8)",
                        }),
        row=4, col=1
        ) 
    f = np.abs(J_np*d2wdt2_phy_hat_np + b_np*dwdt_phy_hat_np - Kt_np*i_phy_hat_np + W_pert_np)**2
    f_mean = np.mean( f )
    fig.add_trace(
        go.Scatter( x=t_physics_np, 
                    y=f, 
                    mode='lines', name = f"mean(|f|^2) = {f_mean:.5f}",
                    line={
                            "color":"rgba( 255, 25, 0, 0.8)",
                        }),
        row=5, col=1
        ) 
    fig.add_hline(y=f_mean, line_dash="dot", row=5, col=1)

    # ------------------------------------------ g physics ------------------------------------------
    fig.add_trace(
        go.Scatter( x=t_physics_np, 
                    y=didt_phy_hat_np, 
                    mode='lines', name = f"L = {L_np:.3f}",
                    line={
                            "color":"rgba( 255, 25, 0, 0.8)",
                        }),
        row=1, col=2
        )   
    fig.add_trace(
        go.Scatter( x=t_physics_np, 
                    y=dwdt_phy_hat_np, 
                    mode='lines', name = f"Ke = {Ke_np:.3f}",
                    line={
                            "color":"rgba( 255, 25, 0, 0.8)",
                        }),
        row=2, col=2
        )  
    fig.add_trace(
        go.Scatter( x=t_physics_np, 
                    y=i_phy_hat_np, 
                    mode='lines', name = f"R = {R_np:.3f}",
                    line={
                            "color":"rgba( 255, 25, 0, 0.8)",
                        }),
        row=3, col=2
        ) 
    fig.add_trace(
        go.Scatter( x=t_physics_np, 
                    y=V_in_np, 
                    mode='lines', name = r"$V_{input}$",
                    line={
                            "color":"rgba( 255, 25, 0, 0.8)",
                        }),
        row=4, col=2
        ) 
    g = np.abs(L_np*didt_phy_hat_np + Ke_np*dwdt_phy_hat_np + R_np*i_phy_hat_np - V_in_np)**2
    g_mean = np.mean( g )
    fig.add_trace(
        go.Scatter( x=t_physics_np, 
                    y=g, 
                    mode='lines', name = f"mean(|g|^2) = {g_mean:.5f}",
                    line={
                            "color":"rgba( 255, 25, 0, 0.8)",
                        }),
        row=5, col=2
        ) 
    fig.add_hline(y=g_mean, line_dash="dot", row=5, col=2)


    # ------------------------------------------ Annotations ------------------------------------------
    texts = [   f"<b>J = {J_np:.3f}</b>", 
                f"<b>L = {L_np:.3f}</b>",
                f"<b>b = {b_np:.3f}</b>", 
                f"<b>Ke = {Ke_np:.3f}</b>",
                f"<b>Kt = {Kt_np:.3f}</b>",
                f"<b>R = {R_np:.3f}</b>",
                "---",
                "---",
                f"<b>mean(|f|^2) = {f_mean:.5f}</b>",
                f"<b>mean(|g|^2) = {g_mean:.5f}</b>"  ]
    for i in range(10):
        if i == 0:
            xr = "x domain"
            yr = "y domain"
        else:
            xr = f"x{i+1} domain"
            yr = f"y{i+1} domain"

        fig.add_annotation(x=0.95,
                           y=0.0,
                           xref=xr,
                           yref=yr,
                           font=dict(
                                    family="Courier New, monospace",
                                    size=15,
                                    color="#020E84"
                                    ),
                           text=texts[i],
                           showarrow=False,
                           align='center')

    fig.update_layout(  height=2500, width=2000, 
                        title_text=f"Final iteration analysis", 
                        showlegend=False)
    fig.write_html(os.path.join(SAVE_DIR,f"physics_analysis.html"), include_mathjax='cdn')