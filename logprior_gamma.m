function logprior = logprior_gamma(y, nu_1, nu_2, inverse)
    
    if inverse
        logprior = -(nu_1 - 1) * log(y) - nu_2 / y;
    else
        logprior = (nu_1 - 1) * log(y) - nu_2 * y;
    end

end