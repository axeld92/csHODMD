%% Compressed Sensing Higher Order DMD (csHODMD)
% This code implements compressed sensing with HODMD for flow field reconstruction
% Author: [Your Name]
% Date: [Current Date]

%% Initialize workspace and parameters
clear; clc;

% Analysis parameters
compression_ratio = 0.05;  % Fraction of points to measure (tunable)
d = 4;                     % Number of HODMD levels
tolerance = 1E-5;         % Tolerance for HODMD
dt = 2/30;               % Time step between snapshots

%% Load and prepare data
% Run readmats to get X, x, y, and other variables
run('readmats.m')

[n, m] = size(X);

%% Compressed sensing setup
[n, m] = size(X);
p = round(n*compression_ratio);
C = zeros(p,n);

% Create random measurement matrix
rng default
perms = randi(n,[p,1]);
for ii = 1:p
    C(ii,perms(ii)) = 1;
end

% Generate compressed measurements
Y = C*X;
Time = (0:m-1)*dt;

%% Perform HODMD on both original and compressed data
% Original data analysis
[PhiX, EigenvaluesX, GrowthRateX, FrequencyX, AmplitudeX] = HODMD(X, d, tolerance, tolerance, dt);
Xrec = reconstructTimeDynamics(PhiX, EigenvaluesX, AmplitudeX, dt, m);
% Compressed data analysis
[PhiY, EigenvaluesY, GrowthRateY, FrequencyY, AmplitudeY] = HODMD(Y, d, tolerance, tolerance, dt);

%% Sparse reconstruction
% Setup sparsifying basis
Psi = fft(eye(n, n));
Theta = C*Psi;
[~,k] = size(PhiY);

% Reconstruct modes using CoSaMP
PhiS = zeros(size(PhiX));
for ii = 1:k
    PhiS(:,ii) = CoSaMP(Theta,PhiY(:,ii),50);
end

% Reconstruct full modes
PhiXrec = Psi*PhiS;

%% Time dynamics reconstruction
time_dynamics = zeros(k, length(Time));
for iter = 1:length(Time)
    time_dynamics(:,iter) = (AmplitudeY.*exp((GrowthRateY + 1i*FrequencyY)*Time(iter)));
end
Xcsrec = PhiXrec * time_dynamics;

% Calculate reconstruction error
recError = norm(X-Xcsrec,"fro")/(numel(X));
fprintf('Reconstruction Error: %e\n', recError);

%% Visualization functions follow...
% [Previous visualization code remains the same]

snapshot = 100;
climits = [min(X(:,snapshot)),max(X(:,snapshot))];
figure
subplot(3,1,1)
contourf(x,y,reshape(real(X(:,snapshot)),mm,nn)','LineStyle','none')
title('Original Snapshot')
axis equal
colorbar
caxis manual
caxis(climits)
subplot(3,1,2)
contourf(x,y,reshape(real(Xrec(:,snapshot)),mm,nn)','LineStyle','none')
title('Reconstructed Snapshot - HODMD')
axis equal
caxis manual
caxis(climits)
colorbar
subplot(3,1,3)
contourf(x,y,reshape(real(Xcsrec(:,snapshot)),mm,nn)','LineStyle','none')
title('Reconstructed Snapshot - csHODMD')
axis equal
colorbar
caxis manual
caxis(climits)
%%
a = [1:length(GrowthRateX)]'; b = num2str(a); c = cellstr(b);
dx = 0; dy = 0.1;
figure

semilogy(FrequencyX,AmplitudeX,'ob',FrequencyY,AmplitudeY,'xk','LineWidth',1);
text(FrequencyX',AmplitudeX', c,'Color','b')
text(FrequencyY',AmplitudeY', c,'Color','k')
set(gca, 'YScale','log')
%hold off
legend('Original modes','Reconstructed modes')

%%
a = [1:length(GrowthRateX)]'; b = num2str(a); c = cellstr(b);
dx = 0; dy = 0.1; % displacement so the text does not overlay the data points
%text(x+dx, y+dy, c);

figure
%hold on
plot(FrequencyX,GrowthRateX,'ob')
hold on
plot(FrequencyY,GrowthRateY,'xk','LineWidth',1);
text(FrequencyX',GrowthRateX', c,'Color','b')
text(FrequencyY',GrowthRateY', c,'Color','k')
legend('Original modes','Reconstructed modes')
xlabel('\omega_i')
ylabel('\delta_i')
ylim([-1,0.5])
hold off

%%
cosines = zeros(k,k);
for ii = 1:k
    for jj = 1:k
        cosines(ii,jj) = norm( dot( PhiX(:,ii) , PhiXrec(:,jj) ) ) / (norm(PhiX(:,ii))*norm(PhiXrec(:,jj)));
    end
end

[frows,fcols] = find(cosines>0.9);
foundModes = [frows fcols];

%%
%mm = 128; nn = 128;

% climits = [min(real(PhiX(:,1))),max(real(PhiX(:,1)))];
%climits = [0,2.5];
hfig = figure;
subplot(3,2,1)
contourf(x,y,reshape(real(PhiX(:,frows(1))),mm,nn)','LineStyle','none')
title('Original Mode')
axis equal
colorbar
climits = [min(real(PhiX(:,frows(1)))),max(real(PhiX(:,frows(1))))];
caxis manual
caxis(climits)
subplot(3,2,2)
contourf(x,y,reshape(real(PhiXrec(:,fcols(1))),mm,nn)','LineStyle','none')
title('Reconstructed Mode')
axis equal
colorbar
caxis manual
caxis(climits)
subplot(3,2,3)
contourf(x,y,reshape(real(PhiX(:,frows(2))),mm,nn)','LineStyle','none')
axis equal
colorbar
climits = [min(real(PhiX(:,frows(2)))),max(real(PhiX(:,frows(2))))];
caxis manual
caxis(climits)
subplot(3,2,4)
contourf(x,y,reshape(real(PhiXrec(:,fcols(2))),mm,nn)','LineStyle','none')
axis equal
colorbar
caxis manual
caxis(climits)
subplot(3,2,5)
contourf(x,y,reshape(real(PhiX(:,frows(4))),mm,nn)','LineStyle','none')
axis equal
colorbar
climits = [min(real(PhiX(:,frows(4)))),max(real(PhiX(:,frows(4))))];
caxis manual
caxis(climits)
subplot(3,2,6)
contourf(x,y,reshape(real(PhiXrec(:,fcols(4))),mm,nn)','LineStyle','none')
axis equal
colorbar
caxis manual
caxis(climits)

set(hfig,'Units','Inches');
pos = get(hfig,'Position');
set(hfig,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])