"""AS3179 Chapter 8 — Dataset 1: Monte Carlo theo mô hình Exponential.

Cài thư viện một lần: python -m pip install numpy scipy pandas matplotlib
Chạy phân tích và xem hình: python data1.py
Chỉ in bảng số, không mở hình: python data1.py --no-plots
Kiểm tra công thức và chạy hai lần cùng seed: python data1.py --validate
Mô hình giả định: Exponential(loc=0, scale=mean(x)). Không kiểm định phân phối ở đây.
Phần kiểm tra phân phối chạy riêng: python check_exponential.py
Không tạo file kết quả hoặc báo cáo.
"""
import argparse
import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
if '--no-plots' in sys.argv or '--validate' in sys.argv:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED, ALPHA, B_MAIN, R_MC = 3179, .05, 10000, 2000
B_VALUES = [100, 1000, 10000, 100000]
MODELS = ['Exponential']
UNIT = 'original time unit (not specified in slide 5)'
TABLE_DATA = {}
FIGURE_NAMES = []
NO_PLOTS = False


def data():
    """Dữ liệu duy nhất: ví dụ (i), slide 5/71, không có thứ tự thời gian gốc."""
    return np.array([3, 5, 7, 18, 43, 85, 91, 98, 100, 130, 230, 487.])


def save_table(name, rows):
    """Giữ bảng trong bộ nhớ; không ghi tệp."""
    df = pd.DataFrame(rows)
    assert np.isfinite(df.select_dtypes(include='number').to_numpy()).all(), name
    assert not df.isna().any().any(), name
    TABLE_DATA[name] = df
    return df


def figure(name):
    """Giữ hình để xem ở cuối chương trình, hoặc đóng khi chạy --no-plots."""
    plt.tight_layout()
    FIGURE_NAMES.append(name)
    if NO_PLOTS:
        plt.close()
    else:
        plt.gcf().set_label(name)


def load(name):
    return TABLE_DATA[name].copy()


def rng(*key):
    return np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=tuple(key)))


def binomial_mcse(p, b):
    return float(np.sqrt(p*(1-p)/b))


def exponential_model():
    """Tham số đầu vào cho mô phỏng; không thực hiện kiểm định độ phù hợp."""
    return stats.expon(loc=0, scale=data().mean())


# PHẦN C: ĐỊNH NGHĨA 1, ĐỊNH LÝ 1 VÀ MỆNH ĐỀ 1


def monte_carlo():
    convergence, verification, selection, probability, paths = [], [], [], [], []
    for j, (model, dist) in enumerate([('Exponential', exponential_model())]):
        mu, variance = dist.mean(), dist.var()
        draws = dist.rvs(size=max(B_VALUES), random_state=rng(2,j,0))
        for b in B_VALUES:
            z = draws[:b]
            mean, sd = z.mean(), z.std(ddof=1)
            se = sd/np.sqrt(b)
            convergence.append(dict(model=model, B=b, model_mean=mu, mean_MC=mean, sd_h=sd,
                MCSE=se, theoretical_MCSE=np.sqrt(variance/b), absolute_error=abs(mean-mu),
                ci_lower=mean-1.96*se, ci_upper=mean+1.96*se, unit=UNIT))
        # Independent paths; nested prefixes within each path visualize the strong LLN.
        checkpoints = np.unique(np.geomspace(10, max(B_VALUES), 150).astype(int))
        plt.figure(figsize=(7,5))
        for path in range(3):
            z = dist.rvs(size=max(B_VALUES), random_state=rng(2,j,1,path))
            cumulative = np.cumsum(z)
            means = cumulative[checkpoints-1]/checkpoints
            plt.plot(checkpoints, means, alpha=.7, label=f'Path {path+1}')
            paths.extend(dict(model=model,path=path+1,B=int(b),mean_MC=a,model_mean=mu)
                         for b,a in zip(checkpoints,means))
        plt.axhline(mu,color='black',linestyle='--',label='Model expectation')
        plt.xscale('log'); plt.xlabel('B'); plt.ylabel('Monte Carlo mean')
        plt.title(model+': convergence paths'); plt.legend(); figure('mc_paths_'+model.lower())
        errors = []
        for b in B_VALUES[:-1]:
            gen = rng(2,j,2,b)
            estimates, ses, sample_vars = [], [], []
            # Bounded memory, 2000 independent experiments (not 2000 prefixes).
            for start in range(0, R_MC, 100):
                z = dist.rvs(size=(min(100,R_MC-start),b),random_state=gen)
                estimates.extend(z.mean(axis=1)); sample_vars.extend(z.var(axis=1,ddof=1))
                ses.extend(z.std(axis=1,ddof=1)/np.sqrt(b))
            estimates, ses = np.array(estimates), np.array(ses)
            empvar = estimates.var(ddof=1)
            coverage = np.mean(abs(estimates-mu) <= 1.96*ses)
            verification.append(dict(model=model,B=b,R=R_MC,model_mean=mu,model_variance=variance,
                mean_estimates=estimates.mean(),bias_MC=estimates.mean()-mu,
                bias_MCSE=estimates.std(ddof=1)/np.sqrt(R_MC), empirical_variance=empvar,
                theoretical_variance=variance/b, B_times_empirical_variance=b*empvar,
                empirical_MCSE=np.sqrt(empvar), theoretical_MCSE=np.sqrt(variance/b),
                mean_sample_variance=np.mean(sample_vars), mean_estimated_MCSE=ses.mean(),
                coverage=coverage,coverage_MCSE=binomial_mcse(coverage,R_MC),unit=UNIT))
            errors.append(estimates-mu)
        plt.figure(figsize=(7,5))
        plt.boxplot(errors, tick_labels=[str(b) for b in B_VALUES[:-1]],showfliers=False)
        plt.axhline(0,color='black',linestyle='--'); plt.xlabel('B'); plt.ylabel('Mean estimate - model expectation')
        plt.title(model+': error distributions (2000 independent runs)')
        figure('mc_error_shrinkage_'+model.lower())
        for delta in [5.,2.,1.]:
            b = int(np.ceil(variance/delta**2))
            for rep in range(5):
                z = dist.rvs(size=b,random_state=rng(2,j,3,int(delta),rep))
                selection.append(dict(model=model,target_MCSE=delta,B_min=b,replicate=rep+1,
                    estimated_MCSE=z.std(ddof=1)/np.sqrt(b),theoretical_MCSE=np.sqrt(variance/b),unit=UNIT))
        p = float(dist.sf(200))
        for mode, target, b in [('main',0.,B_MAIN)]+[
                ('absolute',delta,int(np.ceil(1/(4*delta**2)))) for delta in [.02,.01,.005]]+[
                ('relative',c,int(np.ceil((1-p)/(p*c*c)))) for c in [.10,.05,.02]]:
            z = dist.rvs(size=b,random_state=rng(2,j,4,len(probability)))
            phat = np.mean(z >= 200)
            se = binomial_mcse(phat,b)
            probability.append(dict(model=model,mode=mode,target=target,B=b,threshold=200,
                p_model=p,p_hat=phat,MCSE=se,theoretical_MCSE=binomial_mcse(p,b),
                absolute_error=abs(phat-p),RMCSE=se/p,
                theoretical_RMCSE=np.sqrt((1-p)/(p*b)),bernoulli_variance=p*(1-p),
                worst_case_variance=.25,worst_case_MCSE=1/(2*np.sqrt(b)),unit='probability'))
    save_table('monte_carlo_convergence',convergence)
    df = save_table('mc_theorem_verification',verification)
    save_table('mc_sample_size_selection',selection)
    save_table('mc_probability',probability)
    save_table('mc_convergence_paths',paths)
    slopes = []
    for model in MODELS:
        d = df[df.model == model]
        slope = np.polyfit(np.log(d.B),np.log(d.empirical_MCSE),1)[0]
        slopes.append(dict(model=model,empirical_log_log_slope=slope,theoretical_slope=-.5))
    save_table('mc_mcse_slopes',slopes)
    for column, theory, name, label in [
            ('empirical_MCSE','theoretical_MCSE','mc_mcse','MCSE'),
            ('empirical_variance','theoretical_variance','mc_variance','Variance of Monte Carlo mean')]:
        plt.figure(figsize=(7,5))
        for model in MODELS:
            d = df[df.model == model]
            line, = plt.loglog(d.B,d[column],'o-',label=model+' empirical')
            plt.loglog(d.B,d[theory],'--',color=line.get_color(),label=model+' theory')
        plt.xlabel('B'); plt.ylabel(label); plt.title('Monte Carlo error scaling'); plt.legend()
        figure(name)
    print('02: completed Monte Carlo verification for the Exponential scenario',flush=True)


# PHẦN D: NGHIÊN CỨU MÔ PHỎNG ADEMP


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
    for j,(model,dist) in enumerate([('Exponential', exponential_model())]):
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
            # Asymptotic exponential Wald interval for the corrected median.
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
    save_table('simulation_median_theory',theory)
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
    print('03: completed ADEMP study: Exponential x 3 n x 10000 datasets',flush=True)


# KIỂM TRA CÔNG THỨC VÀ TÁI LẬP, KHÔNG CẦN FILE validate.py.
def check_statistics():
    x = data()
    assert len(x) == 12 and x.max() == 487
    for name, frame in TABLE_DATA.items():
        assert not frame.isna().any().any(), name
        assert np.isfinite(frame.select_dtypes(include='number').to_numpy()).all(), name
    theta = x.mean()
    assert np.isclose(exponential_model().mean(), theta)
    assert np.isclose(exponential_model().var(), theta**2)
    theorem = load('mc_theorem_verification')
    assert len(theorem) == 3 and (theorem.R == R_MC).all()
    assert np.allclose(theorem.theoretical_variance, theorem.model_variance/theorem.B)
    assert np.allclose(theorem.bias_MCSE, np.sqrt(theorem.empirical_variance/R_MC))
    assert np.allclose(theorem.B_times_empirical_variance, theorem.B*theorem.empirical_variance)
    assert np.allclose(theorem.coverage_MCSE,
                       np.sqrt(theorem.coverage*(1-theorem.coverage)/R_MC))
    selection = load('mc_sample_size_selection')
    for row in selection.itertuples():
        variance = theta**2
        assert row.B_min == np.ceil(variance/row.target_MCSE**2)
        assert row.theoretical_MCSE <= row.target_MCSE*(1+1e-10)
    probability = load('mc_probability')
    assert (probability.bernoulli_variance <= .25).all()
    assert np.allclose(probability.RMCSE, probability.MCSE/probability.p_model)
    assert np.allclose(probability.theoretical_RMCSE,
                       np.sqrt((1-probability.p_model)/(probability.p_model*probability.B)))
    metrics = load('simulation_bias_mse')
    assert len(metrics) == 6 and (metrics.B == B_MAIN).all()
    assert np.allclose(metrics.MSE, metrics.bias**2+(B_MAIN-1)/B_MAIN*metrics.EmpSE**2)
    assert np.allclose(metrics.bias_MCSE, metrics.EmpSE/np.sqrt(B_MAIN))
    assert np.allclose(metrics.RMSE**2, metrics.MSE)
    assert metrics.mse_identity_error.max() < 1e-8
    # Đối chiếu độc lập bằng kỳ vọng thống kê thứ tự dưới Exponential.
    for row in metrics[(metrics.model == 'Exponential') &
                       (metrics.method == 'Median / ln(2)')].itertuples():
        n = row.n
        mean_order_lower = theta*np.sum(1/np.arange(n-n//2+1, n+1))
        mean_order_upper = theta*np.sum(1/np.arange(n-(n//2+1)+1, n+1))
        exact_bias = (mean_order_lower+mean_order_upper)/(2*np.log(2))-theta
        assert abs(row.bias-exact_bias) < 5*row.bias_MCSE
    coverage = load('simulation_coverage')
    assert np.allclose(coverage.coverage+coverage.rejection_under_H0, 1)
    comparison = load('simulation_method_comparison')
    assert np.allclose(comparison.empirical_variance_delta, comparison.variance_delta_formula)
    assert np.allclose(comparison.paired_MCSE,
                       np.sqrt(comparison.empirical_variance_delta/B_MAIN))
    check_power_results(TABLE_DATA)
    assert len(TABLE_DATA) == 14 and len(FIGURE_NAMES) == 8
    assert all((frame.model == 'Exponential').all() for frame in TABLE_DATA.values())


def power_row(scenario,test,n1,n2,pvalues,truth,null_value,**extra):
    # Fixed H0 is supplied independently of the data-generating truth.
    b = len(pvalues)
    count = int(np.sum(pvalues < ALPHA))
    rate = count/b
    z = stats.norm.ppf(.975)
    denominator = 1+z*z/b
    center = (rate+z*z/(2*b))/denominator
    half = z*np.sqrt(rate*(1-rate)/b+z*z/(4*b*b))/denominator
    quantity = 'type_I_error' if np.isclose(truth,null_value) else 'power'
    return dict(scenario=scenario,test=test,n1=n1,n2=n2,B=b,alpha=ALPHA,
        H0_value=null_value,truth=truth,effect=truth-null_value,quantity=quantity,
        rejected_count=count,rejection_rate=rate,MCSE=np.sqrt(rate*(1-rate)/b),
        mc_ci_lower=max(0.,center-half),mc_ci_upper=min(1.,center+half),**extra)


def check_power_results(tables):
    power = tables['simulation_power']
    control = tables['simulation_power_null_control']
    assert (power.quantity == 'power').all()
    assert not np.isclose(power.truth,power.H0_value).any()
    assert (control.quantity == 'type_I_error').all()
    assert np.isclose(control.truth,control.H0_value).all()
    for frame in [power,control]:
        assert frame.H0_value.nunique() == 1
        assert (frame.B == B_MAIN).all() and (frame.alpha == ALPHA).all()
        assert np.allclose(frame.rejection_rate,frame.rejected_count/frame.B)
        assert np.allclose(frame.MCSE,np.sqrt(frame.rejection_rate*(1-frame.rejection_rate)/frame.B))
        assert ((frame.mc_ci_lower >= 0)&(frame.mc_ci_upper <= 1)).all()
        assert ((frame.mc_ci_lower <= frame.rejection_rate+1e-12)&
                (frame.rejection_rate <= frame.mc_ci_upper+1e-12)).all()


def plot_power_results(power,control,xcolumn,xlabel,finish):
    rates = pd.concat([power,control],ignore_index=True)
    scenarios = list(rates.scenario.unique())
    tests = list(rates.test.unique())
    fig,axes = plt.subplots(len(scenarios),len(tests),
        figsize=(5*len(tests),3.6*len(scenarios)),squeeze=False)
    for row,scenario in enumerate(scenarios):
        for col,test in enumerate(tests):
            ax = axes[row,col]
            selected = rates[(rates.scenario==scenario)&(rates.test==test)]
            for (n1,n2),values in selected.groupby(['n1','n2']):
                values = values.sort_values(xcolumn)
                label = f'n={n1}' if n2==0 else f'n=({n1},{n2})'
                error = np.vstack([np.maximum(0,values.rejection_rate-values.mc_ci_lower),
                                   np.maximum(0,values.mc_ci_upper-values.rejection_rate)])
                ax.errorbar(values[xcolumn],values.rejection_rate,yerr=error,marker='o',
                            capsize=2,label=label)
            ax.axhline(ALPHA,color='black',linestyle='--',label='alpha=0.05')
            ax.set_title(scenario+' / '+test,fontsize=8)
            ax.set_xlabel(xlabel); ax.set_ylabel('Rejection rate: H0 size / H1 power')
            ax.set_ylim(0,1.02); ax.legend(fontsize=7)
    finish('simulation_power')


def run_analysis():
    TABLE_DATA.clear()
    FIGURE_NAMES.clear()
    print(f'Exponential scenario: theta={data().mean():.6g}, lambda={1/data().mean():.8g}', flush=True)
    print('1/4: Monte Carlo theory verification.', flush=True)
    monte_carlo()
    print('2/4: ADEMP simulation study.', flush=True)
    simulation_study()
    print('3/4: power under fixed-H0 alternatives.', flush=True)
    power_study()
    print('4/4: Formula checks.', flush=True)
    check_statistics()


def print_results():
    """In các bảng tính toán để người dùng tự viết báo cáo."""
    raw_replicates = {'mc_convergence_paths'}
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
    parser = argparse.ArgumentParser(description='Dataset 1: Monte Carlo under an assumed Exponential model')
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
        print(f'VALIDATION PASSED: {len(TABLE_DATA)} tables identical over two full runs.', flush=True)
    print_results()
    print(f'Python {sys.version.split()[0]}; NumPy {np.__version__}; '
          f'Pandas {pd.__version__}; Matplotlib {matplotlib.__version__}; seed={SEED}.', flush=True)
    print('No result files or folders were created. All calculations are in data1.py.', flush=True)
    if not NO_PLOTS:
        print('Close the Matplotlib windows to finish.', flush=True)
        plt.show()


def power_study():
    # H0: mu=theta0 cố định. H1: mu=ratio*theta0; tuyệt đối không thay H0 bằng mu_H1.
    theta0 = data().mean()
    ratios = [.75,1.,1.25,1.5,2.]
    save_table('power_design',[dict(model='Exponential',H0_mean=theta0,
        alternative_mean_ratios='0.75,1,1.25,1.5,2',sample_sizes='12,20,40',B=B_MAIN,
        alpha=ALPHA,alternative='two-sided',
        tests='Student t; exact Exponential chi-square; corrected-median Wald',
        interpretation='ratio=1: Type I control; other ratios: power')])
    rows,reference = [],[]
    for index,ratio in enumerate(ratios):
        mu = ratio*theta0
        for n in [12,20,40]:
            samples = rng(30,index,n).exponential(mu,size=(B_MAIN,n))
            means = samples.mean(axis=1)
            sample_sd = samples.std(axis=1,ddof=1)
            median = np.median(samples,axis=1)/np.log(2)
            p_t = 2*stats.t.sf(abs((means-theta0)/(sample_sd/np.sqrt(n))),n-1)
            pivot = 2*n*means/theta0
            p_exact = np.minimum(1.,2*np.minimum(stats.chi2.cdf(pivot,2*n),stats.chi2.sf(pivot,2*n)))
            median_se = median/(np.log(2)*np.sqrt(n))
            p_median = 2*stats.norm.sf(abs((median-theta0)/median_se))
            for name,pvalues in [('Student t',p_t),('Exact Exponential',p_exact),('Median Wald',p_median)]:
                rows.append(power_row('Exponential',name,n,0,pvalues,mu,theta0,
                    model='Exponential',alternative_ratio=ratio,unit='original time unit'))
            # Đối chiếu độc lập: dưới H1, pivot_H0 = ratio * chi-square(2n).
            critical = stats.chi2.ppf([ALPHA/2,1-ALPHA/2],2*n)
            exact_rate = stats.chi2.cdf(critical[0]/ratio,2*n)+stats.chi2.sf(critical[1]/ratio,2*n)
            simulated_rate = np.mean(p_exact < ALPHA)
            reference.append(dict(model='Exponential',n=n,alternative_ratio=ratio,H0_mean=theta0,
                alternative_mean=mu,quantity='type_I_error' if ratio == 1 else 'power',
                exact_rejection_probability=exact_rate,simulated_rejection_rate=simulated_rate,
                MCSE=np.sqrt(exact_rate*(1-exact_rate)/B_MAIN),
                difference=simulated_rate-exact_rate))
            if ratio == 1:
                assert np.isclose(exact_rate,ALPHA)
    power = save_table('simulation_power',[row for row in rows if row['quantity'] == 'power'])
    control = save_table('simulation_power_null_control',[row for row in rows if row['quantity'] == 'type_I_error'])
    save_table('simulation_power_exact_reference',reference)
    plot_power_results(power,control,'alternative_ratio','Alternative mean / fixed H0 mean',figure)


if __name__ == '__main__':
    main()

