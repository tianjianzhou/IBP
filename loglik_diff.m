function result = loglik_diff(X, Sigma0, Sigma1, inv_Sigma0, inv_Sigma1, logdet_Sigma0, logdet_Sigma1) 

%the ratio of two joint multinormal density, whose latent feature matrices are Z0 and Z1

% return: loglikelihood of (X, Sigma1) - loglikelihood of (X, Sigma0)

    n = size(X, 1);
    inv_Sigma_diff = inv_Sigma1 - inv_Sigma0;
    result = - (1./2) * loglik_quadratic(X, inv_Sigma_diff) - (n/2) * (logdet_Sigma1 - logdet_Sigma0);

end


