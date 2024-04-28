clear all;
clc
% Reading data

Data20F     = readtable('Data.csv');
startDate   = datenum('08-03-2017');  % This is the first day in our sample
endDate     = datenum('01-31-2023');  % This is the last day in our sample
xData       = linspace(startDate, endDate, height(Data20F(end-1382:end,1)));
xData       = datenum(xData);

%%
benchmark    = table2array(Data20F(end-1382:end,2));
factors      = table2array(Data20F(end-1382:end,4:23));

%%
equity_factors = factors(:, 1:7);
currency_factors = factors(:, 8:14);
crypto_factors = factors(:, 15:20);

%%
equity_factor_names = {'SMB';
                       'HML';
                       'RMV';
                       'CMA';
                       'MOM';
                       'ST REV';
                       'LT REV'};

crypto_factor_names = {'Twitter followers';
                       'Marketcap';
                       'BM';
                       'Max30';
                       'r30 0';
                       'Rvol30'};

currency_factor_names = {'CAR';
                         'MOM1';
                         'MOM3';
                         'MOM6';
                         'MOM12';
                         'VOL';
                         'VRP'};


% Computing key metrics

function calculate_metrics(rp)
    
    MN    = mean(rp(:)) * 252;
    SD    = std(rp(:)) *sqrt(252);
    SR    = (mean(rp(:))/std(rp(:))) *sqrt(252);
    KT    = kurtosis(rp(:));
    SW    = skewness(rp(:));
    MDD   = maxdrawdown(exp(cumsum(log(1+rp(:)))));

    cumulative_rp = exp(cumsum(log(1+rp)));
    cumulative_max = cummax(cumulative_rp);
    drawdowns = (cumulative_max - cumulative_rp) ./ cumulative_max;
    UI = sqrt(mean(drawdowns .^ 2));

    fprintf('Mean = %4.3f \n', MN)
    fprintf('Std = %4.3f \n', SD)
    fprintf('Sharpe ratio = %4.3f \n', SR)
    %fprintf('Kurtosis = %4.3f \n', KT)
    %fprintf('Skewness = %4.3f \n', SW)
    fprintf('MDD = %4.3f \n', MDD)
    fprintf('UI = %4.3f \n', UI * sqrt(252))

end


% Plotting comulative returns

function plot_comulative_returns(xData, rp, factor_names)

    factor_num = size(factor_names, 1);
    figure;
    for i = 1:factor_num
        plot(xData(:), exp(cumsum(log(1+rp(:,i)))), 'LineWidth', 1.5);
        hold on
    end
    
    datetick('x', 'yyyy', 'keeplimits')
    recessionplot
    axis([xData(1) xData(end) -inf inf])
    xlabel('Years');
    ylabel('Wealth of an investor, $')
    title(append('Commulative return'))
    legend(factor_names, 'Location', 'Northwest', 'FontSize', 8)

end


% Equity

for nf = 1:size(equity_factors, 2)
    fprintf(equity_factor_names{nf})
    calculate_metrics(equity_factors(:, nf));
end
plot_comulative_returns(xData, equity_factors, equity_factor_names);


% Currency

for nf = 1:size(currency_factors, 2)
    fprintf(currency_factor_names{nf})
    calculate_metrics(currency_factors(:, nf));
end
plot_comulative_returns(xData, currency_factors, currency_factor_names);


% Cryptocurrency

for nf = 1:size(crypto_factors, 2)
    fprintf(crypto_factor_names{nf})
    calculate_metrics(crypto_factors(:, nf));
end
plot_comulative_returns(xData, crypto_factors, crypto_factor_names);


% Equity market

calculate_metrics(benchmark);
plot_comulative_returns(xData, benchmark, 'MKT');


% Construction of the correlation matrix

hold off
corr_matrix = corr([benchmark, factors]);
h = heatmap(corr_matrix);
h.XDisplayLabels = ['MKT'; equity_factor_names; currency_factor_names; crypto_factor_names];
h.YDisplayLabels = ['MKT'; equity_factor_names; currency_factor_names; crypto_factor_names];
colormap('jet');
colormap(flipud(colormap));
clim([-1 1]);
