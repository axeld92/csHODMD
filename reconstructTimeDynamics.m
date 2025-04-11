function Xrec = reconstructTimeDynamics(Modes, Eigenvalues, Amplitudes, dt, K)
    % reconstructTimeDynamics - Reconstruct data from DMD/HODMD components
    %
    % Inputs:
    %   Modes        - DMD modes (J x M)
    %   Eigenvalues  - Eigenvalues (M x 1)
    %   Amplitudes   - Mode amplitudes (M x 1)
    %   dt           - Time step between snapshots
    %   K            - Number of snapshots
    %
    % Output:
    %   Xrec         - Reconstructed data matrix (J x K)

    %M = length(Eigenvalues); % number of modes
    t = (0:K-1) * dt; % time vector

    % Build the Vandermonde matrix for dynamics
    Lambda = exp((log(Eigenvalues(:)) * t)); % M x K

    % Multiply amplitudes and dynamics
    time_dynamics = Amplitudes(:) .* Lambda; % M x K

    % Reconstruct data
    Xrec = Modes * time_dynamics; % J x K
end
