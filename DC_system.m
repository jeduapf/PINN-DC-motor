clear,clc
 
% define the constants
J=0.1;
b=0.5;
Kt=1;
L=1;
R=5;
Ke=1;
 
% define the system matrices
A=[0   1    0; 
   0 -b/J Kt/J; 
   0 -Ke/L -R/L];
B = [0; 
     0; 
     1/L];
W=[0; 
   -1/J; 
   0];

% theta/ omega/ current
C=[0 1 0; 
   0 0 1];
 
% combine the control input and disturbance matrices in a single matrix
Btotal=[B W];
 
% define the state-space model
sys = ss(A,Btotal,C,[]);
 
% discretization constant
dt = 10^-3;
% final simulation time
Tfinal= 5;
% discretized time
time = 0:dt:Tfinal;
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% system response to the step control input 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
% disturbance 
% disturbance = zeros(size(time))';
A = 10^-3;
disturbance = zeros(size(time))';
for f = 1:100
    disturbance = disturbance + A*cos(2*pi*f*rand()*time)';
end
% disturbance = 10^-1*flip(disturbance)/max(disturbance)/2;
disturbance = flip(disturbance);

% control input
figure(1)
control_step = ones(size(time))';
plot(time,control_step,'LineWidth',2)
hold on
input = [control_step,disturbance];
plot(time,input,'LineWidth',2)
xlabel('Time')
ylabel('Control input (voltage)')
title('Control Input Used for Simulation')

% control input and disturbance
U_step=[control_step, disturbance];
 
% simulate the system
output=lsim(sys,U_step,time);
 
% plot the system output 
figure(2)
plot(time,output,'LineWidth',2)
xlabel('Time')
ylabel('System output (angular velocity)')
title('System response to the step control input')

time = time';
save("DC_simulation.mat","time","input","output")