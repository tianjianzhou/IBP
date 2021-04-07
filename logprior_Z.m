function logprior = logprior_Z(Z, alpha)
    
    Z1 = delete_zero(Z);
    [p, K] = size(Z1);
    m = sum(Z1, 1);
    
    H_p = sum(1./(1:p));
    
    if 1 | (sum((p-m) < 0)>0) | (sum((m-1) < 0) > 0)
        g = sprintf('%d ', m);
        fprintf('m: %s\n', g)
    end
    
    logprior = -alpha * H_p + K * log(alpha) - sum(log(1:K)) + sum(arrayfun(@(x) sum(log(1:(x-1))), m)) - sum(arrayfun(@(x) sum(log((p-x+1):p)), m));

end
