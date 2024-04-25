function [time,U_step,output] = DC_motor(J, b, Kt, R, Ke, L, dt, Tfinal,dist_intensity, SNR)
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

    C=[0 1 0; 
       0 0 1];

    % combine the control input and disturbance matrices in a single matrix
    Btotal=[B W];

    % define the state-space model
    sys = ss(A,Btotal,C,[]);
    
    % discretized time
    time = 0:dt:Tfinal;
    %%
    % disturbance 
    disturbance = zeros(size(time))';
    for f = 1:100
        disturbance = disturbance + dist_intensity*cos(2*pi*f*rand()*time)';
    end
    % disturbance = 10^-1*flip(disturbance)/max(disturbance)/2;
    disturbance = flip(disturbance);

    % control input and disturbance
    control = ones(size(time))';
    noise = 10^(log10(max(control))-SNR/20)*rand(size(control));
%     noise = lowpass(noise,180,1/dt);
    U_step=[control + noise, disturbance];

    % simulate the system
    output=lsim(sys,U_step,time);
    time = time';
end