clc;clear;
cvec = (0:0.01:1)';
Cmap = [cvec, cvec, ones(size(cvec));ones(30,3);ones(size(cvec)), cvec(end:-1:1), cvec(end:-1:1)];
name = 'datostuboPablo';
load(fullfile('data', [name, '.mat'])); % load the data

X = U;
[n,m] = size(X);
X = reshape(X,sqrt(n),sqrt(n),m);
X1 = X(:,:,1);
X(:,106:128,:) = [];
X(:,1:16,:) = [];
y(106:128) = [];
y(1:16) = [];
[mm,nn,~]=size(X);
X = reshape(X,mm*nn,m);

[n,m] = size(X); % get the size of each snapshot
p = round(n*0.05);% number of measurements as a fraction of the original number of points
C = zeros(p,n); % Pre-allocate memory for the measurement matrix
rng default % Restart the seed of the random number generator for repeat
perms = randi(n,[p,1]);% create a random vector of p integers from 1 to n
for ii = 1:p
    C(ii,perms(ii)) = 1; %populates each row of C with one element in a random place from perms
end

Y = C*X; % Simulates a measurement of X through C, Y is the matrix of measured snapshots
dt = 2/30;
Time = (0:m-1)*dt; 
d = 4;
e = 1E-5;e1=e;

[Xrec,GrowthRateX,FrequencyX,AmplitudeX,PhiX] =DMDd_SIADS(d,X,Time,e1,e);
[Yrec,GrowthRateY,FrequencyY,AmplitudeY,PhiY] =DMDd_SIADS(d,Y,Time,e1,e);

Psi = fft(eye(n, n)); % sparcifying basis Psi
Theta = C*Psi; % Measure rows of Psi

[~,k] = size(PhiY); 
%%
PhiS = zeros(size(PhiX)); %Pre allocate memory for the sparse modes
tic
for ii = 1:k
ii
PhiS(:,ii) = CoSaMP(Theta,PhiY(:,ii),50); %compute sparse modes
%PhiS(:,ii) = cosamp2(Theta,PhiY(:,ii),75,10^-2,100);
clc
end
toc

PhiXrec = Psi*PhiS; %Reconstruct dense modes from sparse data.

%mm = 128; nn = 128;
t = Time; % time vector
for iter = 1:length(t)
time_dynamics (:,iter) = (AmplitudeY.*exp((GrowthRateY + 1i*FrequencyY)*t(iter)));
end
Xcsrec = PhiXrec * time_dynamics ;

recError = norm(X-Xcsrec,"fro")/(numel(X))

%%
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
% text(FrequencyX',GrowthRateX', c,'Color','b')
% text(FrequencyY',GrowthRateY', c,'Color','k')
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

