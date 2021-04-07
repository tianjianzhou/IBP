function [sigmasq_a_new, Sigma_new, inv_Sigma_new, logdet_Sigma_new] = update_sigmasq_a(X, Z, Sigma, inv_Sigma, logdet_Sigma, sigmasq, sigmasq_a, T, nu_a_1, nu_a_2)
   
    
    p = size(Sigma, 1);
    sigmasq_a_pro = sigmasq_a * exp(normrnd(0, 0.2));
    
    
    Sigma_pro = Sigma + (sigmasq_a_pro - sigmasq_a) * Z * Z';
    inv_Sigma_pro = inv_covmat(Z, sigmasq, sigmasq_a_pro);
    logdet_Sigma_pro = log(det(Sigma_pro));
    
    % loglik_pro - loglik_cur
    loglik_diff1 = loglik_diff(X, Sigma, Sigma_pro, inv_Sigma, inv_Sigma_pro, logdet_Sigma, logdet_Sigma_pro) / T;
    
    % logprior_pro - logprior_cur
    logprior_diff1 = - (nu_a_1 - 1) * log(sigmasq_a_pro) - nu_a_2 / sigmasq_a_pro + (nu_a_1 - 1) * log(sigmasq_a) + nu_a_2 / sigmasq_a;
    
    logJacobian = log(sigmasq_a_pro) - log(sigmasq_a);
    
    u = rand;
    if log(u) < (loglik_diff1 + logprior_diff1 + logJacobian)
        sigmasq_a_new = sigmasq_a_pro;
        Sigma_new = Sigma_pro;
        inv_Sigma_new = inv_Sigma_pro;
        logdet_Sigma_new = logdet_Sigma_pro;
    else
        sigmasq_a_new = sigmasq_a;
        Sigma_new = Sigma;
        inv_Sigma_new = inv_Sigma;
        logdet_Sigma_new = logdet_Sigma;
    end

end
    
    
