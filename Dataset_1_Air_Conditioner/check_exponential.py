"""Dataset 1: thống kê mô tả, MLE và kiểm định độ phù hợp Exponential.


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

SEED, ALPHA, R_KS = 3179, .05, 5000
UNIT = 'original time unit (not specified in slide 5)'


def data():
    # Slide 5/71; thứ tự tăng dần không phải thứ tự thời gian được xác nhận.
    return np.array([3, 5, 7, 18, 43, 85, 91, 98, 100, 130, 230, 487.])


def fitted_exponential(x):
    loc, scale = stats.expon.fit(x, floc=0)
    return stats.expon(loc=loc, scale=scale)


def ks_stat(x, dist):
    ordered = np.sort(x)
    n = len(ordered)
    cdf = dist.cdf(ordered)
    return float(max(np.max(np.arange(1, n+1)/n-cdf),
                     np.max(cdf-np.arange(n)/n)))


def analyze():
    x = data()
    n = len(x)
    q1, q3 = np.quantile(x, [.25, .75], method='linear')
    descriptive = dict(n=n, mean=x.mean(), median=np.median(x),
        variance=x.var(ddof=1), sd=x.std(ddof=1), minimum=x.min(), maximum=x.max(),
        Q1=q1, Q3=q3, IQR=q3-q1, CV=x.std(ddof=1)/x.mean(),
        skewness=stats.skew(x, bias=False))
    dist = fitted_exponential(x)
    scale = dist.mean()
    log_likelihood = dist.logpdf(x).sum()
    observed_d = ks_stat(x, dist)
    # Luồng con cố định để tái lập riêng bước đánh giá phân phối.
    gen = np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=(1, 0)))
    simulated_d = np.empty(R_KS)
    simulated_cv = np.empty(R_KS)
    for r in range(R_KS):
        sample = dist.rvs(size=n, random_state=gen)
        # Bắt buộc khớp lại scale cho từng mẫu; không dùng tham số cố định khi tính D_r.
        refitted = fitted_exponential(sample)
        simulated_d[r] = ks_stat(sample, refitted)
        simulated_cv[r] = sample.std(ddof=1)/sample.mean()
    exceedances = int(np.sum(simulated_d >= observed_d))
    p_value = (1+exceedances)/(R_KS+1)
    q = exceedances/R_KS
    p_mcse = np.sqrt(R_KS*q*(1-q))/(R_KS+1)
    tables = {
        'descriptive_statistics': pd.DataFrame([
            dict(statistic=name, value=value,
                 unit='dimensionless' if name in ['n','CV','skewness'] else
                      'squared original time unit' if name == 'variance' else UNIT)
            for name, value in descriptive.items()]),
        'exponential_fit': pd.DataFrame([dict(model='Exponential', loc=0,
            scale=scale, rate=1/scale, free_parameters=1, log_likelihood=log_likelihood,
            AIC=2-2*log_likelihood, BIC=np.log(n)-2*log_likelihood,
            model_mean=scale, model_variance=scale**2, model_CV=1,
            KS_D=observed_d, KS_R=R_KS, KS_exceedances=exceedances,
            KS_p_corrected=p_value, KS_p_MCSE=p_mcse,
            reject_at_005=bool(p_value < ALPHA), unit=UNIT)]),
        'cv_model_simulation': pd.DataFrame([dict(observed_CV=descriptive['CV'], R=R_KS,
            simulated_CV_mean=simulated_cv.mean(), simulated_CV_q025=np.quantile(simulated_cv,.025),
            simulated_CV_q975=np.quantile(simulated_cv,.975),
            upper_tail_fraction=np.mean(simulated_cv >= descriptive['CV']))]),
        # Chỉ chẩn đoán ảnh hưởng; không bỏ 487 khỏi các phép khớp/kiểm định.
        'maximum_influence': pd.DataFrame([dict(full_mean=x.mean(),
            mean_without_maximum_for_sensitivity_only=x[x != x.max()].mean(),
            max_share_of_total=x.max()/x.sum(), unit=UNIT)])}
    for name, frame in tables.items():
        assert not frame.isna().any().any(), name
        assert np.isfinite(frame.select_dtypes(include='number').to_numpy()).all(), name
    assert np.isclose(scale, x.mean())
    assert np.isclose(observed_d, stats.kstest(x, dist.cdf).statistic)
    assert np.isclose(log_likelihood, -n*(np.log(x.mean())+1))
    return tables, simulated_d


def plot_diagnostics():
    x = data()
    dist = fitted_exponential(x)
    grid = np.linspace(0, x.max()*1.15, 1000)
    plt.figure(figsize=(7,5))
    plt.hist(x, bins='fd', density=True, alpha=.45, label='Observed data (all 12)')
    plt.plot(grid, dist.pdf(grid), label='Fitted Exponential')
    plt.xlabel('Time Between Failures'); plt.ylabel('Density')
    plt.title('Air conditioner failure intervals'); plt.legend(); plt.tight_layout()
    positions = (np.arange(1,len(x)+1)-.5)/len(x)
    theoretical = -dist.mean()*np.log1p(-positions)
    assert np.allclose(theoretical, dist.ppf(positions))
    plt.figure(figsize=(6,5))
    plt.scatter(theoretical, np.sort(x), label='Observed quantiles')
    limit = max(x.max(), theoretical.max())*1.05
    plt.plot([0,limit], [0,limit], '--', label='y = x')
    plt.xlabel('Theoretical quantiles'); plt.ylabel('Observed quantiles')
    plt.title('Exponential Q-Q plot (loc = 0)'); plt.legend(); plt.tight_layout()


def main():
    parser = argparse.ArgumentParser(description='Dataset 1: Exponential goodness-of-fit only')
    parser.add_argument('--no-plots', action='store_true', help='Print tables without opening plots')
    parser.add_argument('--validate', action='store_true', help='Run twice to check reproducibility; no plots')
    args = parser.parse_args()
    print(f'Exponential KS calibration: {R_KS} refitted samples, n={len(data())}.', flush=True)
    tables, simulated_d = analyze()
    if args.validate:
        repeated, repeated_d = analyze()
        assert all(frame.equals(repeated[name]) for name, frame in tables.items())
        assert np.array_equal(simulated_d, repeated_d)
        print('VALIDATION PASSED: all Exponential fit and calibration results reproduced.', flush=True)
    with pd.option_context('display.float_format', lambda value: f'{value:.6g}'):
        for name, frame in tables.items():
            print(f'\n=== {name} ===')
            print(frame.to_string(index=False))
    # H0 là họ Exponential với rate chưa biết. p>alpha không chứng minh H0 đúng.
    rejected = bool(tables['exponential_fit'].reject_at_005.iloc[0])
    print('\nKS decision:', 'reject H0' if rejected else 'do not reject H0 (not proof of the model)')
    if not args.no_plots and not args.validate:
        plot_diagnostics()
        plt.show()


if __name__ == '__main__':
    main()
