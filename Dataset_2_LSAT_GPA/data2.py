"""AS3179 Chapter 8: observed analysis, Monte Carlo and simulation study.

python data2.py
python data2.py --no-plots
python data2.py --validate

All data and code are in this file. No CSV, images or reports are written.
Simulation mechanisms are assumed scenarios, not verified population models.
Lecture references: slides 6-9 (MC), 17-20 (simulation), 52-55 (permutation).
"""
import argparse
import sys
from itertools import combinations
from math import comb
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
if '--no-plots' in sys.argv or '--validate' in sys.argv:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED, ALPHA, B_MAIN, R_MC = 3179, .05, 10000, 2000
B_VALUES = [100, 1000, 10000, 100000]
TABLES, FIGURES = {}, []
NO_PLOTS = False


def rng(*key):
    return np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=tuple(key)))


def store(name, rows):
    frame = pd.DataFrame(rows)
    assert not frame.isna().any().any(), name
    assert np.isfinite(frame.select_dtypes(include='number').to_numpy()).all(), name
    TABLES[name] = frame
    return frame


def finish_figure(name):
    plt.tight_layout()
    FIGURES.append(name)
    if NO_PLOTS:
        plt.close()
    else:
        plt.gcf().set_label(name)


def proportion_se(p, b):
    return float(np.sqrt(p*(1-p)/b))


def permutation_summary(observed, replicates):
    b = len(replicates)
    c = int(np.sum(np.abs(replicates) >= abs(observed)-1e-12))
    q = c/b
    return dict(B=b,exceedances=c,p_corrected=(1+c)/(b+1),
        p_MCSE=np.sqrt(b*q*(1-q))/(b+1),
        tail_ci_lower=0. if c == 0 else stats.beta.ppf(.025,c,b-c+1),
        tail_ci_upper=1. if c == b else stats.beta.ppf(.975,c+1,b-c))


def performance(scenario, n1, n2, method, estimates, truth):
    b = len(estimates)
    bias = estimates.mean()-truth
    sd = estimates.std(ddof=1)
    losses = (estimates-truth)**2
    mse = losses.mean()
    return dict(scenario=scenario,n1=n1,n2=n2,method=method,B=b,truth=truth,
        bias=bias,bias_MCSE=sd/np.sqrt(b),EmpSE=sd,
        EmpSE_MCSE_fourth_moment=np.std((estimates-estimates.mean())**2,ddof=1)/(2*sd*np.sqrt(b)),
        MSE=mse,MSE_MCSE=losses.std(ddof=1)/np.sqrt(b),RMSE=np.sqrt(mse),
        RMSE_MCSE_delta=losses.std(ddof=1)/np.sqrt(b)/(2*np.sqrt(mse)),
        mse_identity_error=abs(mse-bias*bias-(b-1)/b*sd*sd))


def coverage_row(scenario,n1,n2,method,lower,upper,truth):
    hit = (lower <= truth)&(truth <= upper)
    p = hit.mean()
    return dict(scenario=scenario,n1=n1,n2=n2,method=method,B=len(hit),
        H0_value=truth,coverage=p,coverage_MCSE=proportion_se(p,len(hit)),
        rejection_under_true_H0=1-p,rejection_MCSE=proportion_se(p,len(hit)),
        mean_width=np.mean(upper-lower),width_MCSE=np.std(upper-lower,ddof=1)/np.sqrt(len(hit)))


def compare_losses(scenario,n1,n2,t1,t2,independent_t2,truth):
    l1, l2 = (t1-truth)**2, (t2-truth)**2
    delta = l1-l2
    b = len(delta)
    covariance = np.cov(l1,l2,ddof=1)[0,1]
    identity = l1.var(ddof=1)+l2.var(ddof=1)-2*covariance
    se = delta.std(ddof=1)/np.sqrt(b)
    independent = l1-(independent_t2-truth)**2
    return dict(scenario=scenario,n1=n1,n2=n2,B=b,delta_MSE=delta.mean(),
        paired_MCSE=se,ci_lower=delta.mean()-1.96*se,ci_upper=delta.mean()+1.96*se,
        loss_correlation=np.corrcoef(l1,l2)[0,1],variance_delta=delta.var(ddof=1),
        variance_delta_formula=identity,independent_delta_MSE=independent.mean(),
        independent_MCSE=independent.std(ddof=1)/np.sqrt(b),
        unpaired_MCSE_from_marginals=np.sqrt((l1.var(ddof=1)+l2.var(ddof=1))/b))


def monte_carlo():
    # Definitions 1 / Theorem 1 / Proposition 1: draw fresh values from assumed P.
    # This is not resampling from the observed dataset and not a bootstrap SE.
    label, mu, variance, deltas, p_true, draw_h, draw_event = mc_setup()
    convergence, theory, selection, probability = [], [], [], []
    draws = draw_h(rng(10,0),max(B_VALUES))
    for b in B_VALUES:
        sample = draws[:b]
        mean, sd = sample.mean(), sample.std(ddof=1)
        se = sd/np.sqrt(b)
        convergence.append(dict(target=label,B=b,model_expectation=mu,estimate=mean,sd_h=sd,
            MCSE=se,theoretical_MCSE=np.sqrt(variance/b),absolute_error=abs(mean-mu),
            ci_lower=mean-1.96*se,ci_upper=mean+1.96*se))
    checkpoints = np.unique(np.geomspace(10,max(B_VALUES),150).astype(int))
    plt.figure(figsize=(7,5))
    for path in range(3):
        values = draw_h(rng(10,1,path),max(B_VALUES))
        running = np.cumsum(values)[checkpoints-1]/checkpoints
        plt.plot(checkpoints,running,alpha=.7,label=f'Path {path+1}')
    plt.axhline(mu,color='black',linestyle='--',label='Model expectation')
    plt.xscale('log'); plt.xlabel('B'); plt.ylabel(label); plt.legend()
    finish_figure('mc_convergence')
    errors = []
    for b in B_VALUES[:-1]:
        gen = rng(11,b)
        means, ses, sample_vars = [], [], []
        for start in range(0,R_MC,100):
            values = draw_h(gen,(min(100,R_MC-start),b))
            means.extend(values.mean(axis=1))
            sample_vars.extend(values.var(axis=1,ddof=1))
            ses.extend(values.std(axis=1,ddof=1)/np.sqrt(b))
        means, ses = np.asarray(means), np.asarray(ses)
        empirical_variance = means.var(ddof=1)
        cover = np.mean(abs(means-mu) <= 1.96*ses)
        theory.append(dict(B=b,R=R_MC,model_expectation=mu,model_variance=variance,
            bias_MC=means.mean()-mu,bias_MCSE=means.std(ddof=1)/np.sqrt(R_MC),
            empirical_variance=empirical_variance,theoretical_variance=variance/b,
            B_times_empirical_variance=b*empirical_variance,
            empirical_MCSE=np.sqrt(empirical_variance),theoretical_MCSE=np.sqrt(variance/b),
            mean_sample_variance=np.mean(sample_vars),coverage=cover,
            coverage_MCSE=proportion_se(cover,R_MC)))
        errors.append(means-mu)
    plt.figure(figsize=(7,5))
    plt.boxplot(errors,tick_labels=[str(b) for b in B_VALUES[:-1]],showfliers=False)
    plt.axhline(0,color='black',linestyle='--'); plt.xlabel('B'); plt.ylabel('MC error')
    finish_figure('mc_error_shrinkage')
    for index,delta in enumerate(deltas):
        b = int(np.ceil(variance/delta**2))
        for rep in range(5):
            values = draw_h(rng(12,index,rep),b)
            selection.append(dict(target_MCSE=delta,B_min=b,replicate=rep+1,
                estimated_MCSE=values.std(ddof=1)/np.sqrt(b),theoretical_MCSE=np.sqrt(variance/b)))
    designs = [('main',0.,B_MAIN)]
    designs += [('absolute',d,int(np.ceil(1/(4*d*d)))) for d in [.02,.01,.005]]
    designs += [('relative',c,int(np.ceil((1-p_true)/(p_true*c*c)))) for c in [.10,.05,.02]]
    for index,(kind,target,b) in enumerate(designs):
        indicators = draw_event(rng(13,index),b)
        phat = indicators.mean()
        se = proportion_se(phat,b)
        probability.append(dict(kind=kind,target=target,B=b,p_model=p_true,p_hat=phat,
            MCSE=se,theoretical_MCSE=proportion_se(p_true,b),RMCSE=se/p_true,
            theoretical_RMCSE=np.sqrt((1-p_true)/(p_true*b)),
            bernoulli_variance=p_true*(1-p_true),worst_case_MCSE=1/(2*np.sqrt(b))))
    store('monte_carlo_convergence',convergence)
    verified = store('mc_theorem_verification',theory)
    store('mc_sample_size_selection',selection)
    store('mc_probability',probability)
    slope = np.polyfit(np.log(verified.B),np.log(verified.empirical_MCSE),1)[0]
    store('mc_mcse_slope',[dict(empirical_slope=slope,theoretical_slope=-.5)])
    for col,theory_col,name in [('empirical_MCSE','theoretical_MCSE','mc_mcse'),
                              ('empirical_variance','theoretical_variance','mc_variance')]:
        plt.figure(figsize=(7,5))
        plt.loglog(verified.B,verified[col],'o-',label='Empirical')
        plt.loglog(verified.B,verified[theory_col],'--',label='Theory')
        plt.xlabel('B'); plt.ylabel(col); plt.legend(); finish_figure(name)


def study_figures(metrics,coverage):
    # Error bars quantify simulation uncertainty, not observed-data uncertainty.
    for col,se in [('bias','bias_MCSE'),('MSE','MSE_MCSE')]:
        fig,axes = plt.subplots(1,metrics.scenario.nunique(),figsize=(15,4),squeeze=False)
        for ax,(scenario,group) in zip(axes[0],metrics.groupby('scenario')):
            for method,rows in group.groupby('method'):
                ax.errorbar(rows.n1,rows[col],yerr=1.96*rows[se],marker='o',capsize=3,label=method)
            if col == 'bias':
                ax.axhline(0,color='black',linestyle='--')
            ax.set_title(scenario,fontsize=9); ax.set_xlabel('n1'); ax.set_ylabel(col)
            ax.legend(fontsize=7)
        finish_figure('simulation_'+col)
    fig,axes = plt.subplots(1,coverage.scenario.nunique(),figsize=(15,4),squeeze=False)
    for ax,(scenario,group) in zip(axes[0],coverage.groupby('scenario')):
        for method,rows in group.groupby('method'):
            ax.errorbar(rows.n1,rows.coverage,yerr=1.96*rows.coverage_MCSE,marker='o',capsize=3,label=method)
        ax.axhline(.95,color='black',linestyle='--'); ax.set_title(scenario,fontsize=9)
        ax.set_xlabel('n1'); ax.set_ylabel('Coverage of model estimand'); ax.legend(fontsize=7)
    finish_figure('simulation_coverage')


def check_calculations():
    for name,frame in TABLES.items():
        assert not frame.isna().any().any(),name
        assert np.isfinite(frame.select_dtypes(include='number').to_numpy()).all(),name
    v = TABLES['mc_theorem_verification']
    assert len(v)==3 and (v.R==R_MC).all()
    assert np.allclose(v.theoretical_variance,v.model_variance/v.B)
    assert np.allclose(v.bias_MCSE,np.sqrt(v.empirical_variance/R_MC))
    assert np.allclose(v.coverage_MCSE,np.sqrt(v.coverage*(1-v.coverage)/R_MC))
    m = TABLES['simulation_bias_mse']
    assert np.allclose(m.MSE,m.bias**2+(m.B-1)/m.B*m.EmpSE**2)
    assert np.allclose(m.bias_MCSE,m.EmpSE/np.sqrt(m.B))
    assert m.mse_identity_error.max()<1e-8
    c = TABLES['simulation_coverage']
    assert np.allclose(c.coverage+c.rejection_under_true_H0,1)
    assert np.allclose(c.coverage_MCSE,np.sqrt(c.coverage*(1-c.coverage)/c.B))
    d = TABLES['simulation_method_comparison']
    assert np.allclose(d.variance_delta,d.variance_delta_formula)
    assert np.allclose(d.paired_MCSE,np.sqrt(d.variance_delta/d.B))
    p = TABLES['mc_probability']
    assert (p.bernoulli_variance<=.25).all()
    assert np.allclose(p.RMCSE,p.MCSE/p.p_model)
    selected = TABLES['mc_sample_size_selection']
    assert (selected.theoretical_MCSE <= selected.target_MCSE*(1+1e-12)).all()
    check_power_results(TABLES)


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
    TABLES.clear(); FIGURES.clear()
    print('1/4: observed analysis and permutation test.',flush=True)
    observed_analysis()
    print('2/4: Monte Carlo estimation and MCSE verification.',flush=True)
    monte_carlo()
    print('3/4: simulation study.',flush=True)
    simulation_study()
    print('4/4: power under fixed-H0 alternatives and formula checks.',flush=True)
    power_study()
    check_calculations()


def main():
    global NO_PLOTS
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--no-plots',action='store_true',help='Print numerical tables without opening plots')
    parser.add_argument('--validate',action='store_true',help='Run twice and compare all numerical tables')
    args = parser.parse_args()
    NO_PLOTS = args.no_plots or args.validate
    run_analysis()
    if args.validate:
        first = {name:frame.copy(deep=True) for name,frame in TABLES.items()}
        first_figures = FIGURES.copy()
        print('Validation: repeating with the same seed.',flush=True)
        run_analysis()
        assert set(first)==set(TABLES)
        assert all(frame.equals(TABLES[name]) for name,frame in first.items())
        assert first_figures==FIGURES
        print(f'VALIDATION PASSED: {len(TABLES)} numerical tables reproduced exactly.',flush=True)
    with pd.option_context('display.float_format',lambda value:f'{value:.6g}'):
        for name,frame in TABLES.items():
            print(f'\n=== {name} ===')
            print(frame.to_string(index=False))
    print('Simulation DGPs are assumptions; their parameters are not verified population truths.')
    print('MC confidence intervals and sampling confidence intervals are different quantities.')
    print('All results remain in memory; no report or output files were written.')
    if not NO_PLOTS:
        plt.show()


# DATASET 2: mỗi hàng là một cặp LSAT-GPA, không lấy mẫu hai cột độc lập.
def data():
    lsat = np.array([576,635,558,578,666,580,555,661,651,605,653,575,545,572,594.])
    gpa = np.array([3.39,3.30,2.81,3.03,3.44,3.07,3.00,3.43,3.36,3.13,3.12,2.74,2.76,2.88,2.96])
    return lsat, gpa


def model_correlation():
    return float(np.corrcoef(*data())[0,1])


def observed_analysis():
    x, y = data()
    n = len(x)
    assert n == len(y) == 15
    r = model_correlation()
    assert np.isclose(r,stats.pearsonr(x,y).statistic)
    rows = []
    for name,values in [('LSAT',x),('GPA',y)]:
        q1,q3 = np.quantile(values,[.25,.75])
        rows.append(dict(variable=name,n=len(values),mean=values.mean(),median=np.median(values),
            variance=values.var(ddof=1),sd=values.std(ddof=1),minimum=values.min(),maximum=values.max(),
            Q1=q1,Q3=q3,IQR=q3-q1,skewness=stats.skew(values,bias=False)))
    store('observed_statistics',rows)
    xc, yc = x-x.mean(), y-y.mean()
    denominator = np.linalg.norm(xc)*np.linalg.norm(yc)
    gen = np.random.default_rng(SEED)
    permuted = np.array([xc @ gen.permutation(yc)/denominator for _ in range(B_MAIN)])
    result = permutation_summary(r,permuted)
    fisher = np.tanh(np.arctanh(r)+np.array([-1,1])*stats.norm.ppf(.975)/np.sqrt(n-3))
    result.update(n=n,Pearson_r=r,Fisher_ci_lower=fisher[0],Fisher_ci_upper=fisher[1],
        permutation_H0='independence of LSAT and GPA across iid schools',
        Fisher_assumption='iid bivariate normal; approximate interval')
    store('observed_inference',[result])
    # Chưa kiểm chứng chuẩn hai chiều hoặc tính iid từ thiết kế lấy mẫu thật.
    store('working_model_parameters',[dict(mean_LSAT=x.mean(),mean_GPA=y.mean(),
        sd_LSAT=x.std(ddof=1),sd_GPA=y.std(ddof=1),rho=r,
        model_status='assumed DGP, not a verified population distribution')])
    plt.figure(figsize=(7,5)); plt.scatter(x,y)
    plt.xlabel('LSAT'); plt.ylabel('GPA'); plt.title('Observed paired schools')
    finish_figure('observed_scatter')
    plt.figure(figsize=(7,5)); plt.hist(permuted,bins=35,density=True,alpha=.7)
    plt.axvline(abs(r),color='red',linestyle='--'); plt.axvline(-abs(r),color='red',linestyle='--')
    plt.xlabel('Permuted Pearson r'); plt.ylabel('Density'); plt.title('Permutation H0: independence')
    finish_figure('observed_permutation')


def draw_pairs(gen,size,rho,nu=0):
    """Hai biến chuẩn hóa có Var=1, Corr=rho; nu=0: normal, nu>4: Student t."""
    shape = (size,) if isinstance(size,int) else tuple(size)
    u = gen.standard_normal(shape)
    v = rho*u+np.sqrt(1-rho*rho)*gen.standard_normal(shape)
    if nu:
        # Cùng hệ số xuyên tâm cho cả cặp; không phá vỡ cấu trúc phụ thuộc.
        radial = np.sqrt((nu-2)/gen.chisquare(nu,size=shape))
        u, v = u*radial, v*radial
    return u,v


def mc_setup():
    # Mục 1: h(U,V)=UV có E[h]=rho, Var[h]=1+rho^2 dưới chuẩn hai chiều.
    # Đây là ước lượng kỳ vọng của tích chuẩn hóa, không phải sample Pearson r.
    rho = model_correlation()
    threshold = stats.norm.ppf(.95)
    p = stats.norm.sf(threshold)
    store('mc_target',[dict(model='assumed bivariate normal',h='standardized LSAT * standardized GPA',
        expectation=rho,variance=1+rho*rho,probability_event='standardized LSAT > normal 0.95 quantile',
        event_threshold=threshold,p_model=p,unit='dimensionless')])
    def draw_h(gen,size):
        u,v = draw_pairs(gen,size,rho)
        return u*v
    def draw_event(gen,size):
        u,_ = draw_pairs(gen,size,rho)
        return u > threshold
    return 'E[standardized LSAT * standardized GPA]',rho,1+rho*rho,[.05,.02,.01],p,draw_h,draw_event


def correlations(x,y):
    xc = x-x.mean(axis=1,keepdims=True)
    yc = y-y.mean(axis=1,keepdims=True)
    denominator = np.sqrt(np.sum(xc*xc,axis=1)*np.sum(yc*yc,axis=1))
    assert (denominator > 0).all()
    return np.clip(np.sum(xc*yc,axis=1)/denominator,-1+1e-12,1-1e-12)


def correct_correlation(r,n):
    # Hiệu chỉnh Bias bậc một theo mô hình chuẩn, không bảo đảm không chệch hữu hạn.
    return r*(1+(1-r*r)/(2*(n-1)))


def fisher_interval(estimates,n):
    z = np.arctanh(np.clip(estimates,-1+1e-12,1-1e-12))
    half = stats.norm.ppf(.975)/np.sqrt(n-3)
    return np.tanh(z-half),np.tanh(z+half)


def simulation_study():
    # ADEMP (slide 20): estimand=rho; 2 estimators on the SAME B datasets.
    # n1 counts paired schools; n2=0 marks a single paired sample, not two groups.
    # The t8 scenario preserves rho but challenges the normal-based correction and CI.
    rho = model_correlation()
    scenarios = [('normal_fitted_rho',rho,0),('normal_zero_rho',0.,0),('t8_fitted_rho',rho,8)]
    store('simulation_design',[dict(scenario=name,rho=truth,student_df=nu,
        sample_sizes='15,30,60 paired schools',B=B_MAIN,
        estimand='Pearson population correlation',
        methods='Pearson; first-order normal bias correction',
        tests='H0: rho = scenario rho; all rejections are Type I, not power')
        for name,truth,nu in scenarios])
    metrics, coverage, differences, exact_zero = [], [], [], []
    for index,(scenario,truth,nu) in enumerate(scenarios):
        for n in [15,30,60]:
            x,y = draw_pairs(rng(20,index,n,0),(B_MAIN,n),truth,nu)
            r = correlations(x,y)
            corrected = correct_correlation(r,n)
            for method,values in [('Pearson',r),('Pearson first-order corrected',corrected)]:
                metrics.append(performance(scenario,n,0,method,values,truth))
                lower,upper = fisher_interval(values,n)
                row = coverage_row(scenario,n,0,method+' Fisher CI',lower,upper,truth)
                zstat = (np.arctanh(values)-np.arctanh(truth))*np.sqrt(n-3)
                pvals = 2*stats.norm.sf(abs(zstat))
                assert np.isclose(np.mean(pvals < ALPHA),row['rejection_under_true_H0'])
                coverage.append(row)
            # Pearson t test is exact for zero rho under bivariate normal, not under general dependence.
            if nu == 0 and truth == 0:
                tstat = r*np.sqrt((n-2)/(1-r*r))
                rejected = np.mean(2*stats.t.sf(abs(tstat),n-2) < ALPHA)
                exact_zero.append(dict(scenario=scenario,n=n,B=B_MAIN,H0_rho=0,
                    rejection=rejected,rejection_MCSE=proportion_se(rejected,B_MAIN)))
            independent_x,independent_y = draw_pairs(rng(20,index,n,1),(B_MAIN,n),truth,nu)
            independent_corrected = correct_correlation(correlations(independent_x,independent_y),n)
            differences.append(compare_losses(scenario,n,0,r,corrected,independent_corrected,truth))
    m = store('simulation_bias_mse',metrics)
    c = store('simulation_coverage',coverage)
    store('simulation_method_comparison',differences)
    store('simulation_exact_zero_correlation_test',exact_zero)
    study_figures(m,c)


def power_study():
    # H0 cố định rho=0. Zero correlation không đồng nghĩa độc lập dưới Student t.
    # Các p-value dưới đây kiểm định rho=0, không được gọi là p-value hoán vị độc lập.
    null_rho = 0.
    alternatives = [-.6,-.3,0.,.3,.6,model_correlation()]
    families = [('normal',0),('t8_sensitivity',8)]
    store('power_design',[dict(scenario=name,H0_rho=null_rho,student_df=nu,B=B_MAIN,alpha=ALPHA,
        alternative='two-sided',rho_alternatives=str(alternatives),sample_sizes='15,30,60 paired schools',
        tests='Pearson t; Fisher z; first-order corrected Fisher z; no permutation power',
        inference='Pearson t exact only under normal H0; Fisher methods approximate',
        interpretation='rho=0: Type I control; rho != 0: power; t8 checks misspecification')
        for name,nu in families])
    rows = []
    for family,(scenario,nu) in enumerate(families):
        for index,truth in enumerate(alternatives):
            for n in [15,30,60]:
                x,y = draw_pairs(rng(30,family,index,n),(B_MAIN,n),truth,nu)
                r = correlations(x,y)
                corrected = correct_correlation(r,n)
                p_t = 2*stats.t.sf(abs(r)*np.sqrt((n-2)/(1-r*r)),n-2)
                p_fisher = 2*stats.norm.sf(abs(np.arctanh(r)-np.arctanh(null_rho))*np.sqrt(n-3))
                p_corrected = 2*stats.norm.sf(abs(np.arctanh(corrected)-np.arctanh(null_rho))*np.sqrt(n-3))
                for name,pvalues in [('Pearson t',p_t),('Fisher z',p_fisher),('Corrected Fisher z',p_corrected)]:
                    rows.append(power_row(scenario,name,n,0,pvalues,truth,null_rho,
                        student_df=nu,unit='correlation'))
    power = store('simulation_power',[row for row in rows if row['quantity'] == 'power'])
    control = store('simulation_power_null_control',[row for row in rows if row['quantity'] == 'type_I_error'])
    plot_power_results(power,control,'truth','Alternative correlation (fixed H0 rho=0)',finish_figure)


if __name__ == '__main__':
    main()
