clear; clc; close all;

if ~exist('data', 'dir'); mkdir('data'); end
disp('🚀  100 data...');

rng(42); 
file_id = 1;

configs = [
    repmat([8.0, 20.0, 0.5, 2.0], 20, 1);   % high
    repmat([4.0,  8.0, 0.5, 2.0], 20, 1);   % middle
    repmat([1.5,  4.0, 0.5, 1.5], 40, 1);   % low
    repmat([1.0,  1.5, 0.2, 1.0], 20, 1)    % very low
];

for i = 1:100
    h_range = configs(i, 1:2); v_range = configs(i, 3:4);
    h0 = h_range(1) + (h_range(2) - h_range(1)) * rand();
    v0 = v_range(1) + (v_range(2) - v_range(1)) * rand();

    tspan = 0:0.02:7.0; 
    options = odeset('Events', @ground_event); 
    [t, y] = ode45(@drone_dynamics, tspan, [h0, v0], options);
    
    h = y(:, 1); v = y(:, 2);

    % noise
    noise_h = 0.05 * smoothdata(randn(size(h)), 'gaussian', 15); 
    noise_v = 0.10 * smoothdata(randn(size(v)), 'gaussian', 15);
    
    h_noisy = h + noise_h; v_noisy = v + noise_v;

    m = 1.5; g = 9.81; k_drag = 0.8; k_v = 3.0; v_target = 1.5;
    thrust = m*g - k_drag*v_target + k_v .* (v_noisy - v_target); 

    T = table(t, h_noisy, v_noisy, thrust, 'VariableNames', {'time', 'altitude', 'velocity', 'thrust'});
    writetable(T, sprintf('data/landing_%03d.csv', file_id));
    file_id = file_id + 1;
end
disp('🎉 100 data completed!');

function dydt = drone_dynamics(~, y)
    h = y(1); v = y(2); 

    m = 1.5; g = 9.81; k_drag = 0.8; k_v = 3.0; 
    v_target = 1.5; A_ge = 1.0; k_ground = 3.5; alpha = 0.6; epsilon = 0.05;

    dh = -v;
    
    F_drag = k_drag * v; 
    
    v_eff = max(0, v); 
    F_g = A_ge * exp(-k_ground * h) / (h + epsilon) * (1 + alpha * v_eff^2); 
    
    F_thrust = m*g - k_drag*v_target + k_v * (v - v_target); 
    dv = g - (F_thrust + F_drag + F_g) / m;
    dydt = [dh; dv];
end

function [value, isterminal, direction] = ground_event(~, y)
    h = y(1); v = y(2);
    hit_ground = h - 0.02; 
    bounce = v;            

    value = min(hit_ground, bounce); 
    isterminal = 1;      
    direction = -1;      
end