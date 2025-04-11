%Reads the data

%name = 'StaticMixer_FullOpen'; %Change for the name of the desired .mat file
name = 'datostuboPablo';
load(fullfile('data', [name, '.mat'])); % load the data

% U = x velocity
% V = y velocity
% x = x coordinates
% y = y coordinates
%%% Other variables are remnants of the data acquisition and dataset
%%% construction process and can be deleted.


X = U; %create X matrix to avoid overwriting original data
[n,m] = size(X); %get dimensions of data
X = reshape(X,sqrt(n),sqrt(n),m); % reshape as 2D snapshots in time
X1 = X(:,:,1); % Select the first snapshot to trim the zeros

nread = 74; % this line number works good for the static mixer
            % should work fine for the bubble column too

rows = X1(:,nread)~= 0; % get indices for rows
cols = X1(nread,:)~= 0; % and columns that are not equal to zero (that are staying)

X = X(rows,cols,:); % get rid of the rows and columns of zero elements
X1 = X1(rows,cols,:); % This is just to check in the workspace if cleaning worked
x = x(rows); % get the x values of remaining data
y = y(cols); % get the y values of remaining data

%% This is only for the static mixer, can be commented out for the bubble
%% column
% indcs = find(X1 == 0); % find the remaining zero values
% [indcxs, indcys] = ind2sub(size(X1), indcs); % get them as indices
% xwall = x(indcxs); % finally get the x and y indices of
% ywall = y(indcys); % the baffles
%%

[mm,nn,~]=size(X);
X = reshape(X,mm*nn,m); % reshape data as a 2D matrix for analysis

%%% Separate training data (yes) from validation data (not). Can be
%%% commented out
Xnot = X(:,end-9:end); 
Xyes = X(:,1:end-10);