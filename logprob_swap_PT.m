function logp = logprob_swap_PT(X, Sigma1, Sigma2, inv_Sigma1, inv_Sigma2, logdet_Sigma1, logdet_Sigma2, T1, T2)
    logp = (loglik_diff(X, Sigma1, Sigma2, inv_Sigma1, inv_Sigma2, logdet_Sigma1, logdet_Sigma2)) * (1 / T1 - 1 / T2);
end

