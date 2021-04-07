function [sigmasq_new, Sigma_new, inv_Sigma_new, logdet_Sigma_new] = update_sigmasq(X, Z, Sigma, inv_Sigma, logdet_Sigma, sigmasq, sigmasq_a, T, nu_1, nu_2)
   
    
    p = size(Sigma, 1);
    sigmasq_pro = sigmasq * exp(normrnd(0, 0.2));
    
    
    Sigma_pro = Sigma + (sigmasq_pro - sigmasq) * eye(p);
    inv_Sigma_pro = inv_covmat(Z, sigmasq_pro, sigmasq_a);
    logdet_Sigma_pro = log(det(Sigma_pro));
    
    % loglik_pro - loglik_cur
    loglik_diff1 = loglik_diff(X, Sigma, Sigma_pro, inv_Sigma, inv_Sigma_pro, logdet_Sigma, logdet_Sigma_pro) / T;
    
    % logprior_pro - logprior_cur
    logprior_diff1 = - (nu_1 - 1) * log(sigmasq_pro) - nu_2 / sigmasq_pro + (nu_1 - 1) * log(sigmasq) + nu_2 / sigmasq;
    
    logJacobian = log(sigmasq_pro) - log(sigmasq);
    
    u = rand;
    if log(u) < (loglik_diff1 + logprior_diff1 + logJacobian)
        sigmasq_new = sigmasq_pro;
        Sigma_new = Sigma_pro;
        inv_Sigma_new = inv_Sigma_pro;
        logdet_Sigma_new = logdet_Sigma_pro;
    else
        sigmasq_new = sigmasq;
        Sigma_new = Sigma;
        inv_Sigma_new = inv_Sigma;
        logdet_Sigma_new = logdet_Sigma;
    end

end
    
    
