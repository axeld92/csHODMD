function [Modes, Eigenvalues, growth_rates, frequencies, Amplitudes] = cHODMD(X, p, d, epsilon1, epsilon, dt)
    % cHODMD - Perform Compressed Higher Order Dynamic Mode Decomposition (cHODMD).
    %
    % Inputs:
    %   X         - Data matrix (J x K), where each column is a snapshot in time
    %   p         - Desired lower dimension for compression
    %   d         - Parameter d for HODMD (sliding window)
    %   epsilon1  - SVD truncation threshold
    %   epsilon   - Mode amplitude truncation threshold
    %   dt        - Time step between snapshots
    %
    % Outputs:
    %   Modes        - DMD modes (J x M) with unit RMS norm in the original space
    %   Eigenvalues  - Eigenvalues corresponding to each DMD mode
    %   growth_rates - Growth rates (\delta_n)
    %   frequencies  - Frequencies (\omega_n)
    %   Amplitudes   - Real, non-negative mode amplitudes

    % Step 1: Compression
    [J, K] = size(X); % Size of the original data matrix
    C = randn(p, J); % Random compression matrix
    X_compressed = C * X; % Compressed data matrix

    % Step 2: Perform SVD on the compressed data matrix
    [U_compressed, Sigma, T] = svd(X_compressed, 'econ');
    singular_values_squared = diag(Sigma).^2;
    total_energy = sum(singular_values_squared);
    energy = cumsum(singular_values_squared, 'reverse');

    % Determine number of modes N to retain
    N = find(energy / total_energy <= epsilon1, 1, 'last');
    if isempty(N) || N == 0
        N = find(energy / total_energy <= 0.99, 1, 'last');
        if isempty(N) || N == 0
            N = min(size(X_compressed)); % Fallback to all modes
        end
    end

    % Truncate SVD results to N modes
    U_compressed = U_compressed(:, 1:N);
    Sigma = Sigma(1:N, 1:N);
    T = T(:, 1:N);
    V_hat_compressed = Sigma * T'; % Reduced snapshot matrix in the compressed space

    % Step 3: Construct the enlarged-reduced snapshot matrix
    V_tilde_compressed = zeros(N * d, K - d + 1);
    for k = 1:(K - d + 1)
        V_tilde_compressed(:, k) = reshape(V_hat_compressed(:, k:k + d - 1), [], 1);
    end

    % Step 4: Dimension reduction using SVD
    [U_tilde, S_tilde, ~] = svd(V_tilde_compressed, 'econ');
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

    U_tilde = U_tilde(:, 1:hat_N);
    S_tilde = S_tilde(1:hat_N, 1:hat_N);
    bar_V_compressed = S_tilde * U_tilde'; % Reduced-enlarged-reduced snapshot matrix

    % Step 5: Compute the Koopman matrix
    [U3, S1, U4] = svd(bar_V_compressed(:, 1:end-1), 'econ');
    R_reduced = bar_V_compressed(:, 2:end) * U4 * diag(1 ./ diag(S1)) * U3';

    % Step 6: Eigenvalue decomposition
    [bar_Q, D] = eig(R_reduced);
    Eigenvalues = diag(D);
    growth_rates = real(log(Eigenvalues) / dt);
    frequencies = imag(log(Eigenvalues) / dt);

    % Step 7: Compute the enlarged modes in compressed space
    Q_tilde = U_tilde * bar_Q;

    % Step 8: Extract reduced modes and normalize
    Q_hat_compressed = Q_tilde(1:N, :);
    normalized_reduced_modes_compressed = Q_hat_compressed ./ vecnorm(Q_hat_compressed, 2, 1);

    % Step 9: Decompress modes to original space
    Modes_compressed = U_compressed * normalized_reduced_modes_compressed; % Modes in compressed space
    Modes = pinv(C) * Modes_compressed; % Modes in the original space

    % Step 10: Compute amplitudes
    M = diag(Eigenvalues); % Diagonal matrix with eigenvalues
    L = zeros(N * K, length(Eigenvalues)); % Construct matrix L
    for k = 0:(K - 1)
        L((1:N) + k * N, :) = normalized_reduced_modes_compressed * M^k;
    end
    b = V_hat_compressed(:); % Reshape reduced snapshot matrix into column vector
    [U1, S, U2] = svd(L, 'econ');
    S_inv = diag(1 ./ diag(S)); % Inverse of singular values
    Amplitudes = U2 * S_inv * U1' * b;

    % Step 11: Truncate amplitudes
    max_amplitude = max(abs(Amplitudes));
    valid_indices = abs(Amplitudes) / max_amplitude >= epsilon;
    Amplitudes = Amplitudes(valid_indices);
    Eigenvalues = Eigenvalues(valid_indices);
    Modes = Modes(:, valid_indices);
    growth_rates = growth_rates(valid_indices);
    frequencies = frequencies(valid_indices);

    % Step 12: Normalize modes and adjust amplitudes
    for n = 1:length(Amplitudes)
        old_mode = Modes(:, n);
        old_amplitude = Amplitudes(n);
        norm_factor = sqrt(J) * old_amplitude / norm(old_amplitude * old_mode, 2);
        Modes(:, n) = norm_factor * old_mode;
        Amplitudes(n) = norm(old_amplitude * old_mode, 2) / sqrt(J);
    end
end
