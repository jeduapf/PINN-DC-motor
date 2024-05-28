import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from ze.torch_utils import FNN
import torch
from tqdm.auto import tqdm, trange

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

        # Apply "u" to the system
        t, y, _ = sig.lsim(self.sys, inp, t)

        return t,y

class PINN():

    def __init__(self, control_params, system_params, pinn_params):

        # Dashboard parameters
        self.PLOT = control_params['PLOT']
        self.SAVE_DIR = control_params['SAVE_DIR']
        self.SAVE_GIF_DIR = control_params['SAVE_GIF_DIR']
        self.TEXT = control_params['TEXT']
        self.FIGS = control_params['FIGS']
        self.SAVE_FILES = control_params['SAVE_FILES']
        self.SHOW_NET = control_params['SHOW_NET']

        # ---------------------------------------- System ----------------------------------------
        # System parameters
        self.sys_params = system_params
        moteur = motor(self.sys_params)

        # ---------------------------------------- PINN ----------------------------------------
        # PINN construction parameters
        self.DTYPE = pinn_params["DTYPE"]
        self.physics_points = pinn_params["physics_points"]
        self.data_points = pinn_params["data_points"]
        self.validation_points = pinn_params["validation_points"]
        self.neurons = pinn_params["neurons"]
        self.layers = pinn_params["layers"]
        assert isinstance(self.physics_points,int) and isinstance(self.data_points,int) and (self.physics_points<len(self.sys_params["t"]) and self.data_points<len(self.sys_params["t"])), "Amount of data and physics points must be an integer smaller than the total amout of points in the simulation !"
        
        # PINN neural network
        # For this branch I'll implement one single neural network for both outputs
        self.pinn = FNN(N_INPUT = 1, N_OUTPUT = 2, N_HIDDEN = self.neurons, N_LAYERS = self.layers, SHOW = self.SHOW_NET).requires_grad_(True)

        # PINN optimization parameters
        self.learning_rate = pinn_params["learning_rate"]
        self.regularization = pinn_params["regularization"]
        self.epochs = pinn_params["epochs"]
        self.guesses = [torch.nn.Parameter(torch.tensor([float(guess)], requires_grad=True)) for guess in pinn_params["guesses"]]

        # optimizer ADAM
        self.optimiser = torch.optim.Adam( list(self.pinn.parameters()) + self.guesses,lr=self.learning_rate, betas=(0.95, 0.999))
 
        # ---------------------------------------- POINTS ----------------------------------------

        # **************** DATA POINTS ( Don't require grad ) ****************
        self.t_data_pos = np.random.choice(range(len(self.sys_params["t"])), self.data_points, replace=False) 
        self.t_data_np = self.sys_params["t"][self.t_data_pos]
        self.t_data = torch.tensor(self.t_data_np, dtype=self.DTYPE).view(-1,1)

        # y[:,0] Current I_data
        # y[:,1] Speed w_data
        # self.sys_params["motor_inputs"] = [Vin, D]
        _, y = moteur.run(self.sys_params["t"], self.sys_params["motor_inputs"])
        self.I_data_np = y[self.t_data_pos,0]
        self.w_data_np = y[self.t_data_pos,1]
        self.I_data = torch.tensor(self.I_data_np, dtype=self.DTYPE).view(-1,1)
        self.w_data = torch.tensor(self.w_data_np, dtype=self.DTYPE).view(-1,1)

        # **************** PHYSICS POINTS ( Requires grad ) ****************
        self.t_physics_pos = np.linspace(np.min(self.sys_params["t"]), np.max(self.sys_params["t"]), self.physics_points, dtype=int)
        self.t_physics_np = self.sys_params["t"][self.t_physics_pos]
        self.t_physics = torch.tensor(self.t_physics_np, dtype=self.DTYPE).view(-1,1).requires_grad_(True)

        self.V_in = torch.tensor(self.sys_params["motor_inputs"][self.t_physics_pos,0], dtype=self.DTYPE).view(-1,1)
        self.D = torch.tensor(self.sys_params["motor_inputs"][self.t_physics_pos,1], dtype=self.DTYPE).view(-1,1)
        # **************** VALIDATION POINTS ( Don't require grad ) **************** 
        
        self.t_val_pos = np.linspace(np.min(self.sys_params["t"]), np.max(self.sys_params["t"]), self.validation_points, dtype=int)
        self.t_val_np = self.sys_params["t"][self.t_val_pos]
        self.t_val = torch.tensor(self.t_val_np, dtype=self.DTYPE).view(-1,1)

        self.I_val_np = y[self.t_val_pos,0]
        self.w_val_np = y[self.t_val_pos,1]
        self.I_val = torch.tensor(self.I_data_np, dtype=self.DTYPE).view(-1,1)
        self.w_val = torch.tensor(self.w_data_np, dtype=self.DTYPE).view(-1,1)

        # ---------------------------------------- Plotting ----------------------------------------
        # Tracking
        self.constants = []
        self.losses = []
        # self.derivatives = []
        # self.files = []

    def physics_loss(self, t_physics):

        # PHYSICS LOSS
        aux = self.pinn(t_physics)

        I_phy_hat = aux[:,0].view(-1,1)
        w_phy_hat = aux[:,1].view(-1,1)

        I_phy_hat_dt = torch.autograd.grad( outputs=[I_phy_hat],
                                            inputs=[t_physics],
                                            grad_outputs=torch.ones(I_phy_hat.shape),
                                            retain_graph=True, 
                                            create_graph=True)[0].view(-1,1)
        w_phy_hat_dt = torch.autograd.grad( outputs=[w_phy_hat],
                                            inputs=[t_physics],
                                            grad_outputs=torch.ones(w_phy_hat.shape),
                                            retain_graph=True, 
                                            create_graph=True)[0].view(-1,1)

        self.constants.append([guess.item() for guess in self.guesses])

        # system_params = { "J":0.1,    0
        #                   "b":0.5,    1
        #                   "Kt":1,     2
        #                   "L":1,      3
        #                   "R":5,      4
        #                   "Ke":1}     5
        # self.sys_params["motor_inputs"] = [Vin, D]
        f = I_phy_hat_dt + (self.guesses[4]/self.guesses[3])*I_phy_hat + (self.guesses[5]/self.guesses[3])*w_phy_hat - (1/self.guesses[3])*self.V_in
        g = w_phy_hat_dt + (self.guesses[1]/self.guesses[0])*w_phy_hat - (self.guesses[2]/self.guesses[0])*I_phy_hat + (1/self.guesses[0])*self.D

        return torch.mean(f**2), torch.mean(g**2)

    def data_loss(self, t_data):
        # DATA LOSS

        aux =  self.pinn(t_data)
        I_phy_hat = aux[:,0].view(-1,1)
        w_phy_hat = aux[:,1].view(-1,1)

        # if self.SAVE_FILES:
        #     self.f_consts.write(f"{self.constants[-1][0]}, {self.constants[-1][1]}")
        #     self.f_deriv.write(f"{self.derivatives[-1][0]}, {self.derivatives[-1][1]}, {self.derivatives[-1][2]}, {self.derivatives[-1][3]}")

        #     self.derivatives.clear()
        #     self.constants.clear()

        return torch.mean((I_phy_hat - self.I_data)**2), torch.mean((w_phy_hat - self.w_data)**2)

    # def stop(self):
    #     return self.losses[-1][2] < self.stop_eps_u*self.losses[0][2] and torch.abs(self.constants[-1][0]/self.b_torch - 1) < self.stop_eps and torch.abs(self.constants[-1][1]/self.k_torch - 1) < self.stop_eps

    def dashboard(self, i):

        if self.TEXT:
            l1 = 100*self.regularization["lambda_f"]*self.losses[-1][0]
            l2 = 100*self.regularization["lambda_g"]*self.losses[-1][1]
            l3 = 100*self.regularization["lambda_I"]*self.losses[-1][2]
            l4 = 100*self.regularization["lambda_w"]*self.losses[-1][3]

            tqdm.write(f"\n{i}\n>>f_100: {l1:.3f} >>g_100: {l2:.3f} >>I_100: {l3:.3f} >>w_100: {l4:.3f}")
            tqdm.write(f">>f%: {self.losses[0][0]/self.losses[-1][0]:.3f} >>g: {self.losses[0][1]/self.losses[-1][1]:.3f} >>I: {self.losses[0][2]/self.losses[-1][2]:.3f} >>w: {self.losses[0][3]/self.losses[-1][3]:.3f}")
            tqdm.write(f">>J: {self.constants[-1][0]:.3f} >>b: {self.constants[-1][1]:.3f} >>Kt: {self.constants[-1][2]:.3f} >>L: {self.constants[-1][3]:.3f} >>R: {self.constants[-1][4]:.3f} >>Ke: {self.constants[-1][5]:.3f} ")

        # if self.PLOT:
        #     fig = plt.figure(figsize=(12,5))

        #     # To not compute any gradients in this phase
        #     with torch.no_grad():
        #         self.pinn.eval()  # Evaluation mode
        #         u = self.pinn(self.t_test)
        #     self.pinn.train(True) # Back to trainning mode  

        #     plt.scatter(self.t_obs_np, self.u_obs_np, label="Noisy observations", alpha=0.6)
        #     plt.plot(self.t_test_np, u.detach().cpu().numpy(), label="PINN solution", color="tab:green")
        #     plt.title(f"Training step {i}")
        #     plt.legend()

        #     file = os.path.join(self.SAVE_GIF_DIR,"pinn_%.8i.png"%(i+1))
        #     plt.savefig(file, dpi=100, facecolor="white")
        #     self.files.append(file)
        #     plt.close(fig)

    def step(self, i):

        self.optimiser.zero_grad()

        f_loss, g_loss = self.physics_loss(self.t_physics)
        I_loss, w_loss = self.data_loss(self.t_data)

        # auto define the regularization parameters based on the losses of first iteration
        if i == 0 and (self.regularization["lambda_I"] is None or self.regularization["lambda_w"] is None or self.regularization["lambda_f"] is None or self.regularization["lambda_g"] is None): 
            K = torch.log10(I_loss) + torch.log10(w_loss) + torch.log10(f_loss) + torch.log10(g_loss) 
            self.regularization["lambda_I"] = 10**(K.item()-torch.log10(I_loss).item())
            self.regularization["lambda_w"] = 10**(K.item()-torch.log10(w_loss).item())
            self.regularization["lambda_f"] = 10**(K.item()-torch.log10(f_loss).item())
            self.regularization["lambda_g"] = 10**(K.item()-torch.log10(g_loss).item())

        loss =  self.regularization["lambda_I"]*I_loss + self.regularization["lambda_w"]*w_loss
        loss += self.regularization["lambda_f"]*f_loss + self.regularization["lambda_g"]*g_loss

        loss.backward()

        self.optimiser.step()

        # if self.SAVE_FILES : 
        #     self.f_loss.write(f"{phy_loss.item()}, {dat_loss.item()}, {loss.item()}")
        # else:
        self.losses.append([f_loss.item(),
                            g_loss.item(),
                            I_loss.item(),
                            w_loss.item()])

    def train(self):
        self.pinn.train() # Set model to trainning mode

        # if self.SAVE_FILES:
        #     self.SAVE_FILES_LOSSES = os.paht.join(self.SAVE_DIR,"training_losses.txt")
        #     self.f_loss = open(self.SAVE_FILES_LOSSES, "a")

        #     self.SAVE_FILES_CONSTS = os.paht.join(self.SAVE_DIR,"training_constants.txt")
        #     self.f_consts = open(self.SAVE_FILES_CONSTS, "a")

        #     self.SAVE_FILES_DERIV = os.paht.join(self.SAVE_DIR,"training_derivatives.txt")
        #     self.f_deriv = open(self.SAVE_FILES_DERIV, "a")

        if self.FIGS is None:
            self.FIGS = int(self.epochs/100)

        bar = trange(self.epochs)
        for i in bar:
            self.step(i)

            # plot the result as training progresses
            if i % self.FIGS == 0:
                self.dashboard(i)

            # if not self.SAVE_FILES:
            #     # Early stopping in case of convergency
            #     if self.stop():
            #         print("\n\n\t\t Converged, finishing early !\n\n")
            #         break

    def save_plots(self):

        if self.batch is not None:
            p = np.linspace(0, self.physics_points - 1, self.batch, dtype = int)
            force = self.force_np[p]
        else:
             force = self.force_np


        if self.SAVE_FILES:
            self.losses = np.genfromtxt(self.SAVE_FILES_LOSSES, delimiter=",", usemask=True)
            self.constants = np.genfromtxt(self.SAVE_FILES_CONSTS, delimiter=",", usemask=True)
            self.derivatives = np.genfromtxt(self.SAVE_FILES_DERIV, delimiter=",", usemask=True)


        files1, files2 = write_losses(  self.u_obs, self.derivatives, self.constants, 
                                        self.SAVE_DIR, self.losses, force, 
                                        l = [self.regularization_phy,self.regularization_data], 
                                        TEXT = False, PLOT = True, 
                                        fig_pass = self.FIGS, SAVE_PATH = self.SAVE_DIR)

        losses_constants_plot(self.constants, self.losses, self.SAVE_DIR, self.d, self.w0)

        print("\n\nGenerating GIFs...\n\n")
        save_gif_PIL(os.path.join(self.SAVE_DIR,"learning_k_mu.gif"), self.files, fps=60, loop=0)
        save_gif_PIL(os.path.join(self.SAVE_DIR,"loss1.gif"), files1, fps=60, loop=0)
        save_gif_PIL(os.path.join(self.SAVE_DIR,"loss2.gif"), files2, fps=60, loop=0)

    def predict(self, t):
        with torch.no_grad():
            self.pinn.eval()
            return self.pinn(t).detach().cpu().numpy().squeeze()
