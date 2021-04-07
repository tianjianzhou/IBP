function [Z_new, Sigma_new, inv_Sigma_new, logdet_Sigma_new, sigmasq_new, sigmasq_a_new, alpha_new] = update(X, Z, Sigma, inv_Sigma, logdet_Sigma, sigmasq, sigmasq_a, alpha, T, nu_1, nu_2, nu_a_1, nu_a_2, nu_alpha_1, nu_alpha_2)
    
    p = size(X, 2);
    K_max = size(Z, 2);
    
    Z = delete_zero(Z);
    
    for ii = 1:p
        [Z, Sigma, inv_Sigma, logdet_Sigma] = update_Z_j(ii, X, Z, Sigma, inv_Sigma, logdet_Sigma, sigmasq, sigmasq_a, alpha, T);
        % fprintf('Temprature %.2f. The %i-th row finished.\n', T, ii);
    end
    
    K = size(Z, 2);
    alpha = gamrnd(K + nu_alpha_1, 1 / (sum(1./(1:p)) + nu_alpha_2));
    
    [sigmasq, Sigma, inv_Sigma, logdet_Sigma] = update_sigmasq(X, Z, Sigma, inv_Sigma, logdet_Sigma, sigmasq, sigmasq_a, T, nu_1, nu_2);
    [sigmasq_a, Sigma, inv_Sigma, logdet_Sigma] = update_sigmasq_a(X, Z, Sigma, inv_Sigma, logdet_Sigma, sigmasq, sigmasq_a, T, nu_a_1, nu_a_2);
    
    Z_new = add_zero(Z, K_max);
    Sigma_new = Sigma;
    inv_Sigma_new = inv_Sigma;
    logdet_Sigma_new = logdet_Sigma;
    sigmasq_new = sigmasq;
    sigmasq_a_new = sigmasq_a;
    alpha_new = alpha;

end

