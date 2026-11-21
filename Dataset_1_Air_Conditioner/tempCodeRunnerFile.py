def performance(model,n,method,t,theta):
    b = len(t)
    bias, sd = t.mean()-theta, t.std(ddof=1)
    q = (t-theta)**2
    mse = q.mean()
    identity = bias*bias+(b-1)/b*sd*sd
    # Delta-method MCSE using the empirical fourth moment, without assuming normal T.
    mcse_sd = np.std((t-t.mean())**2,ddof=1)/(2*sd*np.sqrt(b))
    return dict(model=model,n=n,B=b,method=method,theta_model=theta,bias=bias,
        bias_MCSE=sd/np.sqrt(b),EmpSE=sd,EmpSE_MCSE_fourth_moment=mcse_sd,
        MSE=mse,MSE_MCSE=q.std(ddof=1)/np.sqrt(b),RMSE=np.sqrt(mse),
        RMSE_MCSE_delta=q.std(ddof=1)/np.sqrt(b)/(2*np.sqrt(mse)),
        mse_identity_rhs=identity,mse_identity_error=abs(mse-identity),
        estimator_skewness=stats.skew(t,bias=False),unit=UNIT)


def interval_row(model,n,name,lower,upper,theta):
    coverage = np.mean((lower <= theta)&(theta <= upper))
    rejection = np.mean((theta < lower)|(theta > upper))
    return dict(model=model,n=n,B=len(lower),interval=name,theta_model=theta,
        coverage=coverage,coverage_MCSE=binomial_mcse(coverage,len(lower)),
        rejection_under_H0=rejection,rejection_MCSE=binomial_mcse(rejection,len(lower)),
        mean_interval_width=np.mean(upper-lower),
        width_MCSE=np.std(upper-lower,ddof=1)/np.sqrt(len(lower)),unit=UNIT)


def simulation_study():
    metrics, coverage_rows, comparisons, theory = [], [], [], []
    for j,(model,dist) in enumerate(fitted_models().items()):
        theta = dist.mean()
        median_limit = dist.median()/np.log(2)
        theory.append(dict(model=model,theta_model=theta,corrected_median_probability_limit=median_limit,
            asymptotic_bias_of_exp_corrected_median=median_limit-theta,
            population_median=dist.median(),unit=UNIT))
        for n in [12,20,40]:
            samples = dist.rvs(size=(B_MAIN,n),random_state=rng(3,j,n,0))
            means = samples.mean(axis=1)
            medians = np.median(samples,axis=1)/np.log(2)
            s = samples.std(axis=1,ddof=1)
            for method,t in [('Mean',means),('Median / ln(2)',medians)]:
                metrics.append(performance(model,n,method,t,theta))
            # t interval and corresponding two-sided t test: identical decisions.
            half = stats.t.ppf(.975,n-1)*s/np.sqrt(n)
            row = interval_row(model,n,'Mean: Student t',means-half,means+half,theta)
            pvals = 2*stats.t.sf(abs(means-theta)/(s/np.sqrt(n)),n-1)
            assert np.isclose(row['rejection_under_H0'],np.mean(pvals < ALPHA))
            coverage_rows.append(row)
            # Explicitly asymptotic exponential Wald interval, misspecified under Weibull.
            half_m = stats.norm.ppf(.975)*medians/(np.log(2)*np.sqrt(n))
            coverage_rows.append(interval_row(model,n,'Median / ln(2): Exponential Wald',
                medians-half_m,medians+half_m,theta))
            if model == 'Exponential':
                lo,hi = 2*n*means/stats.chi2.ppf([.975,.025],2*n)[:,None]
                coverage_rows.append(interval_row(model,n,'Mean: exact Exponential chi-square',lo,hi,theta))
            l1, l2 = (means-theta)**2, (medians-theta)**2
            delta = l1-l2
            # Fresh datasets for method 2, independent of method 1's samples.
            independent = dist.rvs(size=(B_MAIN,n),random_state=rng(3,j,n,1))
            independent_l2 = (np.median(independent,axis=1)/np.log(2)-theta)**2
            independent_delta = l1-independent_l2
            covariance = np.cov(l1,l2,ddof=1)[0,1]
            var_formula = l1.var(ddof=1)+l2.var(ddof=1)-2*covariance
            assert np.isclose(var_formula,delta.var(ddof=1))
            se = delta.std(ddof=1)/np.sqrt(B_MAIN)
            comparisons.append(dict(model=model,n=n,B=B_MAIN,
                MSE_mean_minus_MSE_median=delta.mean(),paired_MCSE=se,
                delta_ci_lower=delta.mean()-1.96*se,delta_ci_upper=delta.mean()+1.96*se,
                loss_correlation=np.corrcoef(l1,l2)[0,1],loss_covariance=covariance,
                empirical_variance_delta=delta.var(ddof=1),variance_delta_formula=var_formula,
                independent_delta=independent_delta.mean(),
                independent_MCSE=independent_delta.std(ddof=1)/np.sqrt(B_MAIN),
                unpaired_MCSE_from_marginals=np.sqrt((l1.var(ddof=1)+l2.var(ddof=1))/B_MAIN),
                unit='squared original time unit'))
    df = save_table('simulation_bias_mse',metrics)
    cov = save_table('simulation_coverage',coverage_rows)
    save_table('simulation_method_comparison',comparisons)
    save_table('simulation_median_misspecification',theory)
    for column,se,name in [('bias','bias_MCSE','simulation_bias'),('MSE','MSE_MCSE','simulation_mse')]:
        plt.figure(figsize=(8,5))
        for (model,method),d in df.groupby(['model','method']):
            plt.errorbar(d.n,d[column],yerr=1.96*d[se],marker='o',capsize=3,label=model+': '+method)
        if column == 'bias':
            plt.axhline(0,color='black',linestyle='--')
        plt.xticks([12,20,40]); plt.xlabel('Sample size n'); plt.ylabel(column+' (error bars: 1.96 MCSE)')
        plt.title('ADEMP: paired estimators, 10000 datasets per scenario'); plt.legend(fontsize=8)
        figure(name)
    plt.figure(figsize=(9,5))
    for (model,interval),d in cov.groupby(['model','interval']):
        plt.errorbar(d.n,d.coverage,yerr=1.96*d.coverage_MCSE,marker='o',capsize=3,label=model+': '+interval)
    plt.axhline(.95,color='black',linestyle='--',label='Nominal 0.95')
    plt.xticks([12,20,40]); plt.xlabel('Sample size n'); plt.ylabel('Coverage (error bars: 1.96 MCSE)')
    plt.title('Coverage of model expectation'); plt.legend(fontsize=7)
    figure('simulation_coverage')
    print('03: completed ADEMP study: 2 models x 3 n x 10000 datasets',flush=True)


# KIỂM TRA CÔNG THỨC VÀ TÁI LẬP, KHÔNG CẦN FILE validate.py.
def check_statistics():
    x = data()
    assert len(x) == 12 and x.max() == 487
    for name, frame in TABLE_DATA.items():
        assert not frame.isna().any().any(), name
        assert np.isfinite(frame.select_dtypes(include='number').to_numpy()).all(), name
    fits = load('distribution_comparison')
    for row in fits.itertuples():
        dist, _, _, count = fit(x, row.model)
        assert row.loc == 0 and row.free_parameters == count
        assert np.isclose(row.KS_D, stats.kstest(x, dist.cdf).statistic)
        assert np.isclose(row.AIC, 2*count-2*row.log_likelihood)
        assert np.isclose(row.BIC, count*np.log(len(x))-2*row.log_likelihood)
        assert np.isclose(row.KS_p_corrected, (1+row.KS_exceedances)/(R_KS+1))
        replicas = load('ks_calibration_replicates')
        replicas = replicas[replicas.model == row.model]
        assert len(replicas) == R_KS
        assert np.sum(replicas.KS_D >= row.KS_D) == row.KS_exceedances
    e = fits[fits.model == 'Exponential'].iloc[0]
    assert np.isclose(e.scale, x.mean())
    assert np.isclose(e.rate, 1/x.mean())
    assert np.isclose(e.model_variance, x.mean()**2)
    assert np.isclose(e.log_likelihood, -len(x)*(np.log(x.mean())+1))
    theorem = load('mc_theorem_verification')
    assert len(theorem) == 6 and (theorem.R == R_MC).all()
    assert np.allclose(theorem.theoretical_variance, theorem.model_variance/theorem.B)
    assert np.allclose(theorem.bias_MCSE, np.sqrt(theorem.empirical_variance/R_MC))
    assert np.allclose(theorem.B_times_empirical_variance, theorem.B*theorem.empirical_variance)
    assert np.allclose(theorem.coverage_MCSE,
                       np.sqrt(theorem.coverage*(1-theorem.coverage)/R_MC))
    selection = load('mc_sample_size_selection')
    for row in selection.itertuples():
        variance = fits[fits.model == row.model].model_variance.iloc[0]
        assert row.B_min == np.ceil(variance/row.target_MCSE**2)
        assert row.theoretical_MCSE <= row.target_MCSE*(1+1e-10)
    probability = load('mc_probability')
    assert (probability.bernoulli_variance <= .25).all()
    assert np.allclose(probability.RMCSE, probability.MCSE/probability.p_model)
    assert np.allclose(probability.theoretical_RMCSE,
                       np.sqrt((1-probability.p_model)/(probability.p_model*probability.B)))
    metrics = load('simulation_bias_mse')
    assert len(metrics) == 12 and (metrics.B == B_MAIN).all()
    assert np.allclose(metrics.MSE, metrics.bias**2+(B_MAIN-1)/B_MAIN*metrics.EmpSE**2)
    assert np.allclose(metrics.bias_MCSE, metrics.EmpSE/np.sqrt(B_MAIN))
    assert np.allclose(metrics.RMSE**2, metrics.MSE)
    assert metrics.mse_identity_error.max() < 1e-8
    # Đối chiếu độc lập bằng kỳ vọng thống kê thứ tự dưới Exponential.
    for row in metrics[(metrics.model == 'Exponential') &
                       (metrics.method == 'Median / ln(2)')].itertuples():
        n = row.n
        mean_order_lower = e.scale*np.sum(1/np.arange(n-n//2+1, n+1))
        mean_order_upper = e.scale*np.sum(1/np.arange(n-(n//2+1)+1, n+1))
        exact_bias = (mean_order_lower+mean_order_upper)/(2*np.log(2))-e.scale
        assert abs(row.bias-exact_bias) < 5*row.bias_MCSE
    coverage = load('simulation_coverage')
    assert np.allclose(coverage.coverage+coverage.rejection_under_H0, 1)
    assert not ((coverage.model == 'Weibull') &
                coverage.interval.str.contains('chi-square')).any()
    comparison = load('simulation_method_comparison')
    assert np.allclose(comparison.empirical_variance_delta, comparison.variance_delta_formula)
    assert np.allclose(comparison.paired_MCSE,
                       np.sqrt(comparison.empirical_variance_delta/B_MAIN))
    assert len(TABLE_DATA) == 15 and len(FIGURE_NAMES) == 13


def run_analysis():
    TABLE_DATA.clear()
    FIGURE_NAMES.clear()
    print('1/4: Fitting Exponential / Weibull; 5000 refitted KS simulations each. Please wait.', flush=True)
    distribution_fitting()
    print('2/4: Monte Carlo theory verification.', flush=True)
    monte_carlo()
    print('3/4: ADEMP simulation study.', flush=True)
    simulation_study()
    print('4/4: Formula checks.', flush=True)
    check_statistics()


def print_results():
    """In các bảng tính toán để người dùng tự viết báo cáo."""
    raw_replicates = {'ks_calibration_replicates', 'mc_convergence_paths'}
    with pd.option_context('display.float_format', lambda value: f'{value:.6g}'):
        for name, frame in TABLE_DATA.items():
            if name in raw_replicates:
                continue
            print(f'\n=== {name} ===', flush=True)
            print(frame.to_string(index=False), flush=True)
    # Bước 16–18: không tự chia nhóm hoặc coi danh sách tăng dần là thứ tự thời gian.
    print('\nPermutation test: not performed; group labels and original time order unavailable.', flush=True)


def main():
    global NO_PLOTS
    parser = argparse.ArgumentParser(description='Dataset 1: all analysis in this single Python file')
    parser.add_argument('--no-plots', action='store_true', help='Print numerical tables without opening plots')
    parser.add_argument('--validate', action='store_true', help='Run twice and check reproducibility; no plots')
    args = parser.parse_args()
    NO_PLOTS = args.no_plots or args.validate
    run_analysis()
    if args.validate:
        first = {name: frame.copy(deep=True) for name, frame in TABLE_DATA.items()}
        first_figures = FIGURE_NAMES.copy()
        print('Validation: running the complete analysis again with the same seed.', flush=True)
        run_analysis()
        assert set(first) == set(TABLE_DATA)
        for name, frame in first.items():
            assert frame.equals(TABLE_DATA[name]), f'Non-reproducible table: {name}'
        assert first_figures == FIGURE_NAMES
        print('VALIDATION PASSED: 15 tables identical over two full runs.', flush=True)
    print_results()
    print(f'Python {sys.version.split()[0]}; NumPy {np.__version__}; '
          f'Pandas {pd.__version__}; Matplotlib {matplotlib.__version__}; seed={SEED}.', flush=True)
    print('No result files or folders were created. All calculations are in data1.py.', flush=True)
    if not NO_PLOTS:
        print('Close the Matplotlib windows to finish.', flush=True)
        plt.show()


if __name__ == '__main__':
    main()

