function inv_Sigma = inv_covmat(Z, sigmasq, sigmasq_a) 
% return: (sigmasq * I + sigmasq_a Z Z')^{-1}
% using Woodbury
% inverting a K x K matrix

    p = size(Z, 1);
    K = size(Z, 2);
    
    inv_Sigma = eye(p) / sigmasq - (1 / sigmasq) * Z * (( (sigmasq / sigmasq_a) * eye(K) + Z' * Z) \ Z');
    
end

