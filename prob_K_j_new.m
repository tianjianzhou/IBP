function [posterior, logdet_Sigma_all] = prob_K_j_new(X, Z, ii, inv_Sigma, logdet_Sigma, sigmasq, sigmasq_a, alpha, T) 
%conditional distribution of the number of new columns added to each row
%given X(observation), latent feature matrix Z and alpha
%Originally the possible outcome could be arbitrarily large, but we truncate
%and sample from the corresponding multinomial
%the truncate level is 4 
    
    K_trunc = 4;
    p = size(Z, 1);
    lambda = alpha / p; 
    loglik = zeros(1, K_trunc);
    logdet_Sigma_all = zeros(1, K_trunc);
    
    for w = 1:K_trunc                    %this is the parameter of the Poisson distribution
                              %this part corresponds to the conditional distribution of X, given Z_{new}
                              %where Z_{new} represents the latent feature matrix after j new active features are
                              %added to the ith row
        %loglik(w) = log(f(X, Z, ii, w, inv_Sigma));
        [loglik(w), logdet_Sigma_all(w)] = loglik_K_j_new(X, Z, ii, w, inv_Sigma, logdet_Sigma, sigmasq, sigmasq_a);
    end
    
    logdet_Sigma_all = [logdet_Sigma, logdet_Sigma_all];
    
    y = 1:K_trunc;
    logprior = log(lambda) * y - log(factorial(y)); %this part corresponds to the prior of the number of new columns
                                %the prior is Poisson distribution is lambda
                                
    logpost = loglik / T + logprior;
    logpost_max = max(logpost);
    
    
    if logpost_max <= -7
        posterior = [1, zeros(1, K_trunc)];
    else
        logpost = logpost - logpost_max;
        posterior = [exp(-logpost_max), exp(logpost)];
        posterior = posterior / sum(posterior);
    end
    
end
