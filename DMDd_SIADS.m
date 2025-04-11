
function  [Vreconst,GrowthRate,Frequency,Amplitude,DMDmode] =DMDd(d,V,Time,varepsilon1,varepsilon)

%%%%%%%%%%%%%%%%%%%%%%%%%  DMD-d %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% %% INPUT: %%
%%% d: parameter of DMD-d (higher order Koopman assumption)
%%% V: snapshot matrix
%%% Time: vector time
%%% varepsilon1: first tolerance (SVD)
%%% varepsilon: second tolerance (DMD-d modes)
%%% %% OUTPUT: %%
%%% Vreconst: reconstruction of the snapshot matrix V
%%% GrowthRate: growht rate of DMD modes
%%% Frequency: frequency of DMD modes(angular frequency)
%%% Amplitude: amplitude of DMD modes
%%% DMDmode: DMD modes
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


[J,K]=size(V);

%% STEP 1: SVD of the original data

[U,Sigma,T]=svd(V,'econ');
sigmas=diag(Sigma);
n=length(sigmas);

NormS=norm(sigmas,2);
kk=0;
for k=1:n
    if norm(sigmas(k:n),2)/NormS>varepsilon1
        kk=kk+1;
    end
end

U=U(:,1:kk);

%% Spatial complexity: kk
('Spatial complexity')
kk

%% Create reduced snapshots matrix
hatT=Sigma(1:kk,1:kk)*T(:,1:kk)';
[N,~]=size(hatT);

%% Create the modified snapshot matrix
tildeT=zeros(d*N,K-d+1);
for ppp=1:d
    tildeT((ppp-1)*N+1:ppp*N,:)=hatT(:,ppp:ppp+K-d);
end

%% Dimension reduction
[U1,Sigma1,T1]=svd(tildeT,'econ');
sigmas1=diag(Sigma1);

Deltat=Time(2)-Time(1);
n=length(sigmas1);

NormS=norm(sigmas1,2);
kk1=0;
for k=1:n
    RRMSEE(k)=norm(sigmas1(k:n),2)/NormS;
    if RRMSEE(k)>varepsilon1
        kk1=kk1+1;
    end
end

('Spatial dimension reduction')
kk1


U1=U1(:,1:kk1);
hatT1=Sigma1(1:kk1,1:kk1)*T1(:,1:kk1)';

%% Reduced modified snapshot matrix
[~,K1]=size(hatT1);
[tildeU1,tildeSigma,tildeU2]=svd(hatT1(:,1:K1-1),'econ');

%% Reduced modified Koopman matrix
tildeR=hatT1(:,2:K1)*tildeU2*pinv(tildeSigma)*tildeU1';
[tildeQ,tildeMM]=eig(tildeR);
eigenvalues=diag(tildeMM);

M=length(eigenvalues);
qq=log(eigenvalues);
GrowthRate=real(qq)/Deltat;
Frequency=imag(qq)/Deltat;

Q=U1*tildeQ;
Q=Q((d-1)*N+1:d*N,:);
[NN,MMM]=size(Q);

for m=1:MMM
    NormQ=Q(:,m);
    Q(:,m)= Q(:,m)/norm(NormQ(:),2);
end

%% Calculate amplitudes
Mm=zeros(NN*K,M);
Bb=zeros(NN*K,1);
aa=eye(MMM);
for k=1:K
    Mm(1+(k-1)*NN:k*NN,:)=Q*aa;
    aa=aa*tildeMM;
    Bb(1+(k-1)*NN:k*NN,1)=hatT(:,k);
end

[Ur,Sigmar,Vr]=svd(Mm,'econ');
a=Vr*(Sigmar\(Ur'*Bb));

u=zeros(NN,M);
for m=1:M
    u(:,m)=a(m)*Q(:,m);
end
Amplitude=zeros(M,1);

for m=1:M
    aca=U*u(:,m);
    Amplitude(m)=norm(aca(:),2)/sqrt(J);
end

UU=[u;GrowthRate';Frequency';Amplitude']';
UU1=sortrows(UU,-(NN+3));

UU=UU1';
u=UU(1:NN,:);
GrowthRate=UU(NN+1,:);
Frequency=UU(NN+2,:);
Amplitude=UU(NN+3,:);
kk3=0;

for m=1:M
    if Amplitude(m)/Amplitude(1)>varepsilon
        kk3=kk3+1;
    else
    end
end

%% Spectral complexity: number of DMD modes.
('Spectral complexity')
kk3
u=u(:,1:kk3);
GrowthRate=GrowthRate(1:kk3);
Frequency=Frequency(1:kk3);
Amplitude=Amplitude(1:kk3);
('Mode number, delta, omega, Amplitude')
GrowthRateOmegAmpl=[(1:kk3)',GrowthRate',Frequency',Amplitude']

%% Reconstruction of the original snapshot matrix
hatTreconst=zeros(N,K);
for k=1:K
    hatTreconst(:,k)= ContReconst_SIADS(Time(k),Time(1),u,GrowthRate,Frequency);
end

Vreconst=U*hatTreconst;
Vreconst = real(Vreconst);

%% Calculation of DMD modes
DMDmode=zeros(J,kk3);
Amplitude0=zeros(kk3,1);
for m=1:kk3
    NormMode=norm(U*u(:,m),2)/sqrt(J);
    Amplitude0(m)=NormMode;
    DMDmode(:,m)=U*u(:,m)/NormMode;
end

%If the calculation of the amplitudes is correct, ErrAmpl=0
%ErrAmpl=norm(Amplitude(:,1:kk3)-Amplitude0',2)



