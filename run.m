
% Running PTMCMC
% First argument: data file, "data.csv"
% Second argument: number of MCMC iterations
[Z_spls, alpha_spls, sigmasq_spls, sigmasq_a_spls, logpost_spls, K_spls] = MCMC_real_data('X.csv', 2);

% Finding MAP estimate
[~, map_index] = max(logpost_spls);
Z_map = delete_zero(Z_spls(:, :, map_index));

sigmasq_map = sigmasq_spls(1, map_index);
sigmasq_a_map = sigmasq_a_spls(1, map_index);

Z_map_lof = left_order(Z_map);

