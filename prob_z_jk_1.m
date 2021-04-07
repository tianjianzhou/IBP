function result = prob_z_jk_1(X, Sigma0, Sigma1, inv_Sigma0, inv_Sigma1, logdet_Sigma0, logdet_Sigma1, Z, ii, k, T) 


% return: p(z_{ii, k} == 1 | ...)
                                                                     
    p = size(Z, 1);
    
    % m_{(-ii), k} = active - Z(ii, k) in the paper
    active = sum(Z(:, k));
    
    % rho = logposterior(z_{ii,k} == 1) - logposterior(z_{ii,k} == 0)
    rho = (loglik_diff(X, Sigma0, Sigma1, inv_Sigma0, inv_Sigma1, logdet_Sigma0, logdet_Sigma1)) / T + log(active - Z(ii, k)) - log(p - active + Z(ii, k));
    
    % a = p(z_{ii,k} == 1 | ...) / p(z_{ii,k} == 0 | ...)
    a = exp(rho);
    % result = p(z_{ii, k} == 1 | ...)
    result = a / (1 + a);
    
end

