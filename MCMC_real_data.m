function [Z_spls, alpha_spls, sigmasq_spls, sigmasq_a_spls, logpost_spls, K_spls] = MCMC_real_data(data_file, niter)
    
    X = csvread(data_file);
    %X = X(1:100, 1:100);
    [n, p] = size(X);
    
    % seed
    rng(100);
    
    % hyperparameters
    nu_1 = 1.; 
    nu_2 = 1.;
    nu_a_1 = 1.;
    nu_a_2 = 1.;
    nu_alpha_1 = 1.;
    nu_alpha_2 = 1.;
    
    % maximal number of columns of Z
    K_max = 80; 
    
    % number of temperatures
    %n_Tmp = 11;
    n_Tmp = 11;
    T0 = 1.2;
    Tmps = T0.^(0:(n_Tmp - 1));
    
    
    % exchange or not??
    exch = zeros(niter, n_Tmp - 1);
    
    Z_Tmps = zeros(p, K_max, n_Tmp);
    Sigma_Tmps = zeros(p, p, n_Tmp);
    inv_Sigma_Tmps = zeros(p, p, n_Tmp);
    logdet_Sigma_Tmps = zeros(1, n_Tmp);
    alpha_Tmps = zeros(1, n_Tmp);
    sigmasq_Tmps = zeros(1, n_Tmp);
    sigmasq_a_Tmps = zeros(1, n_Tmp);
    
    Z_spls = zeros(p, K_max, niter);
    % Sigma_spls = zeros(p, p, niter);
    % inv_Sigma_spls = zeros(p, p, niter);
    alpha_spls = zeros(1, niter);
    sigmasq_spls = zeros(1, niter);
    sigmasq_a_spls = zeros(1, niter);
    % record the number of features at each iteration
    K_spls = zeros(1, niter);
    logpost_spls = zeros(1, niter);
    
    Z_init = binornd(1, 0.5, p, 1);
    sigmasq_init = 0.3;
    sigmasq_a_init = 0.3;
    Sigma_init = sigmasq_a_init * (Z_init * Z_init') + sigmasq_init * eye(p);
    inv_Sigma_init = inv_covmat(Z_init, sigmasq_init, sigmasq_a_init);
    logdet_Sigma_init = log(det(Sigma_init));
    alpha_init = 1.;
    
    parfor q = 1:n_Tmp
        Z_Tmps(:, :, q) = add_zero(Z_init, K_max);
        Sigma_Tmps(:, :, q) = Sigma_init;
        inv_Sigma_Tmps(:, :, q) = inv_Sigma_init;
        logdet_Sigma_Tmps(1, q) = logdet_Sigma_init;
        sigmasq_Tmps(1, q) = sigmasq_init;
        sigmasq_a_Tmps(1, q) = sigmasq_a_init;
        alpha_Tmps(1, q) = alpha_init;
    end
    
    tic;
    
    timer_every = 1;
    %timer_every = floor(niter/5);
    
    fprintf('MCMC has started.\n');
    % start MCMC iteration
    for i1 = 1:niter
        
        parfor q = 1:n_Tmp
            [Z_Tmps(:, :, q), Sigma_Tmps(:, :, q), inv_Sigma_Tmps(:, :, q), ...
                logdet_Sigma_Tmps(1, q), sigmasq_Tmps(1, q), sigmasq_a_Tmps(1, q), ...
                alpha_Tmps(1, q)] = ...
                update(X, Z_Tmps(:, :, q), Sigma_Tmps(:, :, q), ...
                inv_Sigma_Tmps(:, :, q), logdet_Sigma_Tmps(1, q), sigmasq_Tmps(1, q), ...
                sigmasq_a_Tmps(1, q), alpha_Tmps(1, q), Tmps(q), ...
                nu_1, nu_2, nu_a_1, nu_a_2, nu_alpha_1, nu_alpha_2);
        end
        
        for q = (n_Tmp - 1):(-1):1
            % calculate swap probability only by likelihood
            if log(unifrnd(0, 1)) <= logprob_swap_PT(X, Sigma_Tmps(:, :, q), Sigma_Tmps(:, :, q+1), inv_Sigma_Tmps(:, :, q), inv_Sigma_Tmps(:, :, q+1), logdet_Sigma_Tmps(1, q), logdet_Sigma_Tmps(1, q+1), Tmps(q), Tmps(q+1))
                exch(i1, q) = 1;
                
                % swap two tempratures
                Z_Tmps(:, :, [q, q+1]) = Z_Tmps(:, :, [q+1, q]);
                Sigma_Tmps(:, :, [q, q+1]) = Sigma_Tmps(:, :, [q+1, q]);
                inv_Sigma_Tmps(:, :, [q, q+1]) = inv_Sigma_Tmps(:, :, [q+1, q]);
                logdet_Sigma_Tmps(1, [q, q+1]) = logdet_Sigma_Tmps(1, [q+1, q]);
                sigmasq_Tmps(1, [q, q+1]) = sigmasq_Tmps(1, [q+1, q]);
                sigmasq_a_Tmps(1, [q, q+1]) = sigmasq_a_Tmps(1, [q+1, q]);
                alpha_Tmps(1, [q, q+1]) = alpha_Tmps(1, [q+1, q]);
            end
        end
        
        Z_spls(:, :, i1) = Z_Tmps(:, :, 1);
        alpha_spls(1, i1) = alpha_Tmps(1, 1);
        sigmasq_spls(1, i1) = sigmasq_Tmps(1, 1);
        sigmasq_a_spls(1, i1) = sigmasq_a_Tmps(1, 1);
        
        logpost_spls(1, i1) = - (1./2) * loglik_quadratic(X, inv_Sigma_Tmps(:, :, 1)) - ...
            (n/2) * logdet_Sigma_Tmps(1, 1) + ...
            logprior_Z(Z_Tmps(:, :, 1), alpha_Tmps(1, 1)) + ...
            logprior_gamma(alpha_Tmps(1, 1), nu_alpha_1, nu_alpha_2, false) + ...
            logprior_gamma(sigmasq_Tmps(1, 1), nu_1, nu_2, true) + ...
            logprior_gamma(sigmasq_a_Tmps(1, 1), nu_a_1, nu_a_2, true);
                              
        fprintf('sigmasq: %.5f; sigmasq_a: %.5f; logpost: %.2f.\n', sigmasq_spls(1, i1), sigmasq_a_spls(1, i1), logpost_spls(1, i1));
        
        K_spls(1, i1) = size(delete_zero(Z_Tmps(:, :, 1)), 2);
        
        if mod(i1, timer_every) == 0
            fprintf('%.2f%% of MCMC has been done.\n', i1 / niter * 100);
            toc;
        end
    end
    
    fprintf('MCMC has finished.\n');
    %toc;
    
end
