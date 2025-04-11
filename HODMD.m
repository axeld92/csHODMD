function [Modes, Eigenvalues, growth_rates, frequencies, Amplitudes] = HODMD(X, d, epsilon1, epsilon, dt)
    % HODMD - Perform Higher Order Dynamic Mode Decomposition (HODMD) on a data matrix.
    %
    % Inputs:
    %   X         - Data matrix (J x K), where each column is a snapshot in time
    %   d         - Parameter d for HODMD (sliding window)
    %   epsilon1  - SVD truncation threshold
    %   epsilon   - Mode amplitude truncation threshold
    %   dt        - Time step between snapshots
    %
    % Outputs:
    %   Modes        - DMD modes (J x M) with unit RMS norm
    %   Eigenvalues  - Eigenvalues corresponding to each DMD mode
    %   growth_rates - Growth rates (\delta_n)
    %   frequencies  - Frequencies (\omega_n)
    %   Amplitudes   - Real, non-negative mode amplitudes

    fprintf('Starting HODMD analysis...\n');
    fprintf('Input matrix size: [%d x %d]\n', size(X));
    fprintf('Parameters: d=%d, epsilon1=%e, epsilon=%e, dt=%f\n', d, epsilon1, epsilon, dt);

    % Step 1: Perform SVD on the data matrix X
    fprintf('\nStep 1: Performing initial SVD...\n');
    [U, Sigma, T] = svd(X, 'econ');
    [J, K] = size(X); % Define J and K
    R = min(J, K); % Effective rank of X
    
    % Compute the squared singular values and cumulative energy
    singular_values_squared = diag(Sigma).^2;
    total_energy = sum(singular_values_squared);
    energy = cumsum(singular_values_squared, 'reverse');
    
    % Determine number of modes N to retain
    N = find(energy / total_energy <= epsilon1, 1, 'last');
    if isempty(N) || N == 0
        N = find(energy / total_energy <= 0.99, 1, 'last');
        if isempty(N) || N == 0
            N = R; % Fallback to using all available modes
        end
    end
    fprintf('Selected number of modes (N): %d\n', N);
    % Truncate SVD results to N modes
    U = U(:, 1:N);
    Sigma = Sigma(1:N, 1:N);
    T = T(:, 1:N);
    V_hat = Sigma * T'; % Reduced snapshot matrix

    % Step 2: Construct the enlarged-reduced snapshot matrix
    fprintf('\nStep 2: Constructing enlarged snapshot matrix...\n');
    bar_N = N * d; % Size of enlarged snapshots
    V_tilde = zeros(bar_N, K - d + 1); % Enlarged snapshot matrix
    for k = 1:(K - d + 1)
        V_tilde(:, k) = reshape(V_hat(:, k:k + d - 1), [], 1);
    end

    % Step 3: Dimension reduction of V_tilde using SVD
    fprintf('\nStep 3: Performing dimension reduction...\n');
    [U_tilde, S_tilde, ~] = svd(V_tilde, 'econ');
    singular_values_squared = diag(S_tilde).^2;
    total_energy = sum(singular_values_squared);
    energy = cumsum(singular_values_squared, 'reverse');
    
    % Determine number of modes to retain
    hat_N = find(energy / total_energy <= epsilon1, 1, 'last');
    if isempty(hat_N) || hat_N == 0
        hat_N = find(energy / total_energy <= 0.99, 1, 'last');
        if isempty(hat_N) || hat_N == 0
            hat_N = size(S_tilde, 1); % Fallback to all modes
        end
    end
    fprintf('Number of retained modes after reduction (hat_N): %d\n', hat_N);
    U_tilde = U_tilde(:, 1:hat_N);
    S_tilde = S_tilde(1:hat_N, 1:hat_N);
    bar_V = S_tilde * U_tilde'; % Reduced-enlarged-reduced snapshot matrix

    % Step 4: Compute the reduced-enlarged-reduced Koopman matrix
    [U3, S1, U4] = svd(bar_V(:, 1:end-1), 'econ');
    R_reduced = bar_V(:, 2:end) * U4 * diag(1 ./ diag(S1)) * U3';

    % Step 5: Eigenvalue decomposition of the reduced Koopman matrix
    fprintf('\nStep 5: Performing eigenvalue decomposition...\n');
    [bar_Q, D] = eig(R_reduced);
    Eigenvalues = diag(D);
    growth_rates = real(log(Eigenvalues) / dt);
    frequencies = imag(log(Eigenvalues) / dt);

    % Step 6: Compute the enlarged modes
    Q_tilde = U_tilde * bar_Q;

    % Step 7: Compute the reduced modes
    Q_hat = Q_tilde(1:N, :);
    normalized_reduced_modes = Q_hat ./ vecnorm(Q_hat, 2, 1);

    % Step 8: Compute the final DMD modes
    Modes = U * normalized_reduced_modes;

    % Step 9: Compute mode amplitudes
    fprintf('\nStep 9: Computing mode amplitudes...\n');
    M = diag(Eigenvalues); % Diagonal matrix with eigenvalues (\mu_n)
    L = zeros(N * K, length(Eigenvalues)); % Construct matrix L
    for k = 0:(K - 1)
        L((1:N) + k * N, :) = normalized_reduced_modes * M^k;
    end
    b = V_hat(:); % Reshape reduced snapshot matrix into column vector
    [U1, S, U2] = svd(L, 'econ');
    S_inv = diag(1 ./ diag(S)); % Inverse of singular values
    Amplitudes = U2 * S_inv * U1' * b;

    % Step 10: Truncate amplitudes
    fprintf('\nStep 10: Truncating modes based on amplitude...\n');
    max_amplitude = max(abs(Amplitudes));
    valid_indices = abs(Amplitudes) / max_amplitude >= epsilon;
    Amplitudes = Amplitudes(valid_indices);
    Eigenvalues = Eigenvalues(valid_indices);
    Modes = Modes(:, valid_indices);
    growth_rates = growth_rates(valid_indices);
    frequencies = frequencies(valid_indices);

    % Step 11: Normalize modes and adjust amplitudes
    fprintf('\nStep 11: Performing final normalization...\n');
    for n = 1:length(Amplitudes)
        old_mode = Modes(:, n);
        old_amplitude = Amplitudes(n);
        norm_factor = sqrt(J) * old_amplitude / norm(old_amplitude * old_mode, 2);
        Modes(:, n) = norm_factor * old_mode;
        Amplitudes(n) = norm(old_amplitude * old_mode, 2) / sqrt(J);
    end
    fprintf('\nHODMD analysis complete.\n');
    fprintf('Final number of modes: %d\n', length(Amplitudes));
end
