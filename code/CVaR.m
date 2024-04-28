% Computing optimal weights for a CVaR portfolio

function w = CVaR(benchmark, factors, B, beta)
    fun = @(theta) objectivefunc((-1)*(benchmark + factors * theta), beta);
    options = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'sqp');
    nf = size(factors, 2);
    X0  = 0 * ones(nf, 1); 
    A   = ones(1, nf);
    Aeq = [];
    Beq = [];
    LB  = zeros(1, nf);
    UB  = Inf(1, nf);
    w = fmincon(fun, X0, A, B, Aeq, Beq, LB, UB, [], options);
end


% Defining the objective function for a CVaR portfolio

function CVaR_value = objectivefunc(returns, beta)
    sorted = sort(returns);
    n = numel(sorted);
    index = floor((1-beta) * n);
    CVaR_value = mean(sorted(1:index));
end
