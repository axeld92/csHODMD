%%% Run DMDd in matrices
clc; clear

readmats

%function [Vreconst,growthRates,frequencies,amplitudes,Un] = DMDd(V,d,Time,e1,e2)
dt = 1/15;
[~,numSnapshots] = size(Xyes);
Time = 0:dt:dt*numSnapshots;
d = 350;
p = 1000;
e1 = 1E-3;
e2 = e1;
tic
[Modes, Eigenvalues, growth_rates, frequencies, Amplitudes] = HODMD(X, d, e1, e2, dt);

toc
%%
figure;
semilogy(frequencies, Amplitudes, 'bx', 'LineWidth', 1, 'MarkerSize', 4);
xlabel('Frequency ($\omega$)', 'Interpreter', 'latex');
ylabel('Amplitude (log scale)', 'Interpreter', 'latex');
title('DMD Mode Amplitudes vs Frequencies', 'Interpreter', 'latex');
grid on;

figure;
plot(real(Eigenvalues), imag(Eigenvalues), 'bx', 'LineWidth', 1, 'MarkerSize', 4);
xlabel('Real', 'Interpreter', 'latex');
ylabel('Imaginary', 'Interpreter', 'latex');
title('DMD Eigenvalues', 'Interpreter', 'latex');
grid on;