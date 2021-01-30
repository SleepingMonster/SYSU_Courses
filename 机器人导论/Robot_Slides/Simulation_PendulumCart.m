% The program is to stabilize the cart-pendulum system with LQR controller.
% =========================================================================
% =========================================================================
clear all;
m = 0.09;                                           % The pendulum mass: m = 0.09kg;
M = 1.52+0.09;                                  % The mobile cart mass: M = 1.52kg;
L = 0.24;                                            % The 1/2 length of the pendulum: L = 0.24m;
I = m*(2*L)*(2*L)/12;                        % The inertia of the pendulum: I = m*(2L)^2/12;
r = 0.085/2;                                        % The radius of the driver wheel
Ke = 7*1e-3;                                      % Torque constant 扭矩常量
nr = 12;                                              % Velocity ratio of the motor
Ra = 2;                                              % Resistor of the motor  
g = 9.81;                                            % The gravity accelation: g = 9.81 N/m;
Ts = 0.02;                                          % The sampling time; 
% ========= The state space model of the cart-pendulum system ==============
% X_dot(t) = A*X(t)+B*u(t);
% Y(t) = C*X(t)+D*u(t);
% Compute the system parameters of the dynamical model for the cart-pendulum system;
temp_den = (m+M)*I+m*M*L*L; 
a = m*g*L*(m+M)/temp_den; lamda = -m*m*L*L*g/temp_den;
A_m = [0 1 0 0;a 0 0 0;0 0 0 1;lamda 0 0 0]; B_m = [0 -m*L/temp_den 0 (I+m*L*L)/temp_den]';
C_m = [1 0 0 0;0 0 1 0]; D_m =[0 0]';
Ao_pole = eig(A_m);                     % Find the eigenvalues of the system matrix A;
sys_con = ss(A_m,B_m,C_m,D_m);      % ss用来创建连续状态空间模型
[Wn,Z] = damp(sys_con);              % Find the natural frequency of the system（返回系统的固有频率和衰减因子）
x0 =[-0.2 0 0.2 0]';        % 初始状态，可变
% x0 =[0.5 0 0 0]';        % 初始状态，可变
Tf = 4; T = Tf/Ts;          % 采样周期
initial(sys_con,x0);                    % The system response under the initial state x0
kp = 1/(m+M)/g; Ap_square = m*L*g*(m+M)/((m+M)*(I+m*L*L)+m*m*L*L);
systf_num = kp; systf_den = [1/Ap_square 0 -1];
% ============== Consider motor dynamics =============================
alfa = nr*Ke/Ra/r; beita = -Ke*Ke*nr*nr/Ra/r/r;
c = m*L*beita/temp_den;  d = -(I+m*L*L)*beita/temp_den;
gama = -m*L*alfa/temp_den; delta = (I+m*L*L)*alfa/temp_den;
A_m1 = [0 1 0 0;a 0 0 c;0 0 0 1;lamda 0 0 d]; B_m1 = [0 gama 0 delta]';
Ao_pole1 = eig(A_m1);                     % Find the eigenvalues of the system matrix A;
sys_con1 = ss(A_m1,B_m1,C_m,D_m);
[Wn1,Z1] = damp(sys_con1);              % Find the natural frequency of the system
x0 =[-0.2 0 0.2 0]';    % 初始状态
initial(sys_con1,x0);                    % The system response under the initial state x0
figure; step(systf_num,systf_den);
systf_num1 = kp*alfa; systf_den1 = [1/Ap_square kp*beita*r/nr -1];
hold on; step(systf_num1,systf_den1);
close all
% ============Continuous-Time System==============================
% des_P = [-25+j,-25-j;-4+j,-4-j];
% Kf_p = place(A_m,B_m,des_P);
% sys_conc = ss(A_m-B_m*Kf_p,[],C_m-D_m*Kf_p,[]);
% [Wn1,Z1] = damp(sys_conc);
% figure; initial(sys_conc,x0); grid;
% title('State response of the Open System')
% ===============================================================
% Discretize the continuous-time system to obtain a discrte-time system:
% X(k+1) = G*X(k)+H*u(k);
% Y(k) = C*X(k)+D*u(k);
% where G = e^(A_m*Ts), H = int(e^(A_m(Ts-tao))*B_m), C = C_m, D = D_m;
[G,H] = c2d(A_m,B_m,Ts);          % Convert continuous-time system to a discrete-time system 
ploe_sys = eig(G);                         % Find the eigenvalues of the system matrix G; 
% ======== Consider motor dynamics ====================================== 
[G1,H1] = c2d(A_m1,B_m1,Ts);          % Convert continuous-time system to a discrete-time system 
ploe_sys1 = eig(G1);                         % Find the eigenvalues of the system matrix G;
% ================================================================
% The controllablity matrix Mc =[H G*H G^2*H G^3*H];
% If the matrix Mc is full rank, the system is controllable and can be stablized;
Mc = ctrb(A_m,B_m);                   %  The controllability matrix: Mc =[H G*H G^2*H G^3*H];
r_Mc = rank(Mc); 
% =========== Consider motor dynamics===================================
Mc1 = ctrb(A_m1,B_m1);                   %  The controllability matrix: Mc =[H G*H G^2*H G^3*H];
r_Mc1 = rank(Mc1); 
% ====================Feedback Control Laws ===========================
% Apply PD, poles placement and LQR schemes to stabilize the cart-pendulum system
% ====================PD Controller =================================
des_vec = zeros(4,T);       
set_x = []; set_y = []; set_u = []; error_vec = []; error_dvec = []; sum_u = []; sum_u(1) = 0;
set_u(:,1) = zeros(4,1); set_x(:,1) = x0; error_vec(:,1) = x0; error_dvec(:,1) = x0; set_y(:,1) = C_m*set_x(:,1)+D_m*set_u(1);
Kp = 1*[50,10,10,10]; Kd = 0.01*[10,10,10,10];  %PD控制器参数
% Kp = 1*[50,10,10,10]; Kd = 0.0001*[10,10,10,10];  %PD控制器参数
eig_pd = eig(G+H*Kp-H*Kd);
for k = 2:T
    set_x(:,k) = G*set_x(:,k-1)+H*sum_u(k-1); set_y(:,k) = C_m*set_x(:,k-1)+D_m*sum_u(k-1);
    for i = 1:4
        error_vec(i,k) = des_vec(i,k)-set_x(i,k);
        if k <= 30
            error_dvec(i,k) = (error_vec(i,k)-error_vec(i,k-1))/Ts;
        else
            error_dvec(i,k) = 1/2*((set_x(i,k)-set_x(i,k-3))/3+(set_x(i,k-1)-set_x(i,k-2)))/Ts;
        end
        set_u(i,k) = -(Kp(i)*error_vec(i,k)+Kd(i)*error_dvec(i,k));
    end
    sum_u(k) = sum(set_u(:,k));
end
t = 0:Ts:(size(set_y(1,:),2)-1)*Ts;
figure; subplot(4,1,1); plot(t,set_x(1,:)); grid; title('PD angular');
subplot(4,1,2); plot(t,set_x(2,:),'k');  grid; title('angular velocity')
subplot(4,1,3); plot(t,set_x(3,:),'k');  grid; title('cart position')
subplot(4,1,4); plot(t,set_x(4,:),'k');  grid; title('cart velocity')
% ====================Poles Placement  ============================
% des_poles = [0.95-0.009*j, 0.95+0.009*j; 0.955-0.04301*j, 0.955+0.04301*j];   % Ts = 0.01s;
% des_poles = [0.92-0.009*j, 0.92+0.009*j, 0.955-0.04301*j, 0.955+0.04301*j];  % Ts = 0.02s;
des_poles = [0.95-0.009*j, 0.95+0.009*j, 0.95+0.0356i,0.95-0.0356i];  % Ts = 0.04s;
Kf_pp = place(G,H,des_poles);
% ============ Consier motordynamics===============================
Kf_pp1 = place(G1,H1,des_poles);
% === Experimental data=============================================
Kf_exp = [122.5707, 22.1821,58.6808,35.4851];  % Ts = 0.01s;
K_coe = 7;        % The ratio of the computed input u to the actual sent signal to the motor    
% With this feedback control law u(k) = -Kf*X(k), the closed-loop system is hence obtained as:
% X(k) = (G-H*Kf_pp)*X(k-1);
% Y(k) = (C-D*Kf_pp)*X(k);
set_x = []; set_y = []; set_u = []; set_x(:,1) = x0; set_y(:,1) = (C_m-D_m*Kf_pp)*x0; set_upp(1) = -Kf_pp*set_x(:,1);
for k = 2:T
    set_x(:,k) = (G-H*Kf_pp)*set_x(:,k-1);
    set_y(:,k) = (C_m-D_m*Kf_pp)*set_x(:,k);
    set_upp(k) = -Kf_pp*set_x(:,k);
end
eig_pp = eig(G-H*Kf_pp);
eig_pp1 = eig(G1-H1*Kf_pp1);
eig_ppdelay = eig([G,-H*Kf_pp;eye(size(G,1),size(G,2)),zeros(size(G,1),size(G,2))]);
t = 0:Ts:(size(set_y(1,:),2)-1)*Ts;
figure; subplot(4,1,1); plot(t,set_x(1,:)); grid; title('Pole Placement angular');
subplot(4,1,2); plot(t,set_x(2,:),'k');  grid; title('angular velocity')
subplot(4,1,3); plot(t,set_x(3,:),'k');  grid; title('cart position')
subplot(4,1,4); plot(t,set_x(4,:),'k');  grid; title('cart velocity')
figure; plot(t,set_x(1,:)); hold on; plot(t,set_x(4,:),'k');  grid; title('angular versus cart velocity')
legend('Angular','Cart Velocity')
% ============================= LQR =============================
% LQR algorithm is applied to stablize the system. The state-feedback law is: u(k) = -Kf*X(k);
close all
Qr_c = 3*eye(size(A_m,1),size(A_m,1)); Rr_c = 1;          % The weighting matries Qr and Rr for LQR to compute the feedback vector Kf;
Qr_d = 3*eye(size(A_m,1),size(A_m,1)); Rr_d = 3;
Qr_d(1,1) = 10; Qr_d(2,2) = 30; Qr_d(3,3) = 60;  Qr_d(4,4) = 100; 
% [Kf1,S1,E1] = lqr(A_m,B_m,Qr_c,Rr_c);   % Kf is the computed optimal state feedback law Kf, and E is the eigenvalue vector for (A_m-B_m*Kf);
[Kf_lqr,S,E] = dlqr(G,H,Qr_d,Rr_d);               % Kf is the computed optimal state feedback law Kf, and E is the eigenvalue vector for (G-H*Kf);
[Kf_lqr1,S1,E1] = dlqr(G1,H1,Qr_d,Rr_d);
eig_lqrdelay = eig([G,-H*Kf_lqr;eye(size(G,1),size(G,2)),zeros(size(G,1),size(G,2))]);
eig_lqr = eig(G-H*Kf_lqr);
% sys_conc_lqr = ss(A_m-B_m*Kf1,[],C_m-D_m*Kf1,[]);
% [Wn2,Z2] = damp(sys_conc_lqr);
% figure; initial(sys_conc_lqr,x0); grid; title('Continuous LQR');
% sys_cond = ss(G-H*Kf_lqr,[],C_m-D_m*Kf_lqr,[]);
% [Wn3,Z3] = damp(sys_cond);
set_x = []; set_y = []; set_u = []; set_x(:,1) = x0; set_y(:,1) = (C_m-D_m*Kf_lqr)*x0; set_ulqr(1) = -Kf_lqr*set_x(:,1);
for k = 2:T
    set_x(:,k) = (G-H*Kf_lqr)*set_x(:,k-1);
    set_y(:,k) = (C_m-D_m*Kf_lqr)*set_x(:,k);
    set_ulqr(k) = -Kf_lqr*set_x(:,k);
end
t = 0:Ts:(size(set_y(1,:),2)-1)*Ts;
figure; subplot(4,1,1); plot(t,set_x(1,:)); grid; title('Angular');
subplot(4,1,2); plot(t,set_x(2,:),'k');  grid; title('angular velocity')
subplot(4,1,3); plot(t,set_x(3,:),'k');  grid; title('cart position')
subplot(4,1,4); plot(t,set_x(4,:),'k');  grid; title('cart velocity')
figure; plot(t,set_ulqr); hold on; plot(t,set_x(1,:),'k');  grid; title('Torque versus cart position')
figure; plot(t,sum_u,'m'); hold on; plot(t,set_upp,'k');  grid; 
plot(t,set_ulqr,'b'); 
legend('PD','Pole placement','LQR')
title('Control Torque')
disp('PD control Kp, PD control Kp, Pole placement, LQR')
[Kp.', Kd.', Kf_pp.', Kf_lqr.']
disp('open loop eigenvalues, PD control, Pole placement, LQR')
[ploe_sys, eig_pd, eig_pp,eig_lqr]
% S = logm(Zp)/Ts; 1/10*S   % sampling frequency should be 5~10 times of
% the frequency of the principle ploes of the closed-loop systems