function A_hat = estimate_A(X, Z, sigmasq, sigmasq_a)

    n = size(X, 1);
    p = size(Z, 1);
    K = size(Z, 2);
    
    A_hat = zeros(K, n);
    
    parfor i = 1:n
        inv_a_i_post_covmat = (1. / sigmasq_a) * eye(K) + (1. / sigmasq) * (Z' * Z);
        a_i_post_mean = (1. / sigmasq) * (inv_a_i_post_covmat \ (Z' * X(i, :)'));
        A_hat(:, i) = a_i_post_mean;
    end
    
end
