"""AS3179 Chapter 8: observed analysis, Monte Carlo and simulation study.

python data3.py
python data3.py --no-plots
python data3.py --validate

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


# DATASET 3: thời gian sống dương; Gamma chỉ là kịch bản giả định cho mô phỏng.
def data():
    treatment = np.array([94,197,16,38,99,141,23.])
    control = np.array([52,104,146,10,51,30,40,27,46.])
    return treatment,control


def gamma_parameters():
    # Method of moments: shape=mean^2/variance, scale=variance/mean.
    # Không tuyên bố đã kiểm định hai nhóm có phân phối Gamma.
    result = []
    for values in data():
        mean,variance = values.mean(),values.var(ddof=1)
        result.extend([mean*mean/variance,variance/mean])
    return tuple(result)


def observed_analysis():
    treatment,control = data()
    n1,n2 = len(treatment),len(control)
    assert (n1,n2) == (7,9)
    rows = []
    for name,values in [('Treatment',treatment),('Control',control)]:
        q1,q3 = np.quantile(values,[.25,.75])
        rows.append(dict(group=name,n=len(values),mean=values.mean(),median=np.median(values),
            variance=values.var(ddof=1),sd=values.std(ddof=1),minimum=values.min(),maximum=values.max(),
            Q1=q1,Q3=q3,IQR=q3-q1,skewness=stats.skew(values,bias=False),unit='days'))
    store('observed_statistics',rows)
    combined = np.r_[treatment,control]
    observed = treatment.mean()-control.mean()
    total = combined.sum()
    sums = np.array([combined[list(indices)].sum() for indices in combinations(range(n1+n2),n1)])
    exact = sums/n1-(total-sums)/n2
    assert len(exact) == comb(n1+n2,n1)
    exact_c = np.sum(abs(exact) >= abs(observed)-1e-12)
    gen = np.random.default_rng(SEED)
    mc = np.empty(B_MAIN)
    for b in range(B_MAIN):
        s1 = combined[gen.choice(n1+n2,n1,replace=False)].sum()
        mc[b] = s1/n1-(total-s1)/n2
    result = permutation_summary(observed,mc)
    v1,v2 = treatment.var(ddof=1)/n1,control.var(ddof=1)/n2
    dof = (v1+v2)**2/(v1*v1/(n1-1)+v2*v2/(n2-1))
    half = stats.t.ppf(.975,dof)*np.sqrt(v1+v2)
    result.update(n1=n1,n2=n2,mean_difference=observed,exact_assignments=len(exact),
        exact_extreme=int(exact_c),exact_p=exact_c/len(exact),
        permutation_SD=exact.std(ddof=0),
        permutation_SD_formula=combined.std(ddof=1)*np.sqrt(1/n1+1/n2),
        Welch_p=stats.ttest_ind(treatment,control,equal_var=False).pvalue,
        Welch_ci_lower=observed-half,Welch_ci_upper=observed+half,unit='days',
        permutation_H0='equal distributions with exchangeable independent group labels')
    assert np.isclose(result['permutation_SD'],result['permutation_SD_formula'])
    store('observed_inference',[result])
    k1,s1,k2,s2 = gamma_parameters()
    store('working_model_parameters',[dict(group=group,shape=k,scale=s,model_mean=k*s,
        model_variance=k*s*s,status='assumed Gamma scenario; not a verified population model')
        for group,k,s in [('Treatment',k1,s1),('Control',k2,s2)]])
    plt.figure(figsize=(7,5))
    plt.boxplot([treatment,control],tick_labels=['Treatment','Control'])
    for index,values in enumerate([treatment,control],start=1):
        plt.scatter(np.full(len(values),index),values,alpha=.7)
    plt.ylabel('Survival time (days)'); plt.title('All observed mice')
    finish_figure('observed_groups')
    plt.figure(figsize=(7,5)); plt.hist(exact,bins=35,density=True,alpha=.7)
    plt.axvline(abs(observed),color='red',linestyle='--')
    plt.axvline(-abs(observed),color='red',linestyle='--')
    plt.xlabel('Treatment mean - control mean (days)'); plt.ylabel('Density')
    plt.title('Exact permutation distribution: conditional on exchangeability')
    finish_figure('observed_permutation')


def mc_setup():
    # Mục 1: h(T,C)=T-C; hai lần rút độc lập từ các Gamma giả định.
    k1,s1,k2,s2 = gamma_parameters()
    mean = k1*s1-k2*s2
    variance = k1*s1*s1+k2*s2*s2
    # T/s1 ~ Gamma(k1,1), C/s2 ~ Gamma(k2,1).
    # T>=C tương đương U/(U+V)>=s2/(s1+s2); tỷ số có phân phối Beta(k1,k2).
    p = stats.beta.sf(s2/(s1+s2),k1,k2)
    store('mc_target',[dict(model='independent assumed Gamma groups',h='T-C',
        expectation=mean,variance=variance,probability_event='T >= C',p_model=p,unit='days')])
    def draw_h(gen,size):
        return gen.gamma(k1,s1,size=size)-gen.gamma(k2,s2,size=size)
    def draw_event(gen,size):
        return gen.gamma(k1,s1,size=size) >= gen.gamma(k2,s2,size=size)
    return 'E[T-C] (days)',mean,variance,[5.,2.,1.],p,draw_h,draw_event


def group_estimates(treatment,control,k1,k2):
    n1,n2 = treatment.shape[1],control.shape[1]
    means = treatment.mean(axis=1)-control.mean(axis=1)
    v1,v2 = treatment.var(axis=1,ddof=1),control.var(axis=1,ddof=1)
    a,b = v1/n1,v2/n2
    welch_se = np.sqrt(a+b)
    welch_df = (a+b)**2/(a*a/(n1-1)+b*b/(n2-1))
    welch_half = stats.t.ppf(.975,welch_df)*welch_se
    pooled_variance = ((n1-1)*v1+(n2-1)*v2)/(n1+n2-2)
    pooled_se = np.sqrt(pooled_variance*(1/n1+1/n2))
    pooled_half = stats.t.ppf(.975,n1+n2-2)*pooled_se
    q1,q2 = stats.gamma.ppf(.5,[k1,k2])
    median1,median2 = np.median(treatment,axis=1),np.median(control,axis=1)
    factor1,factor2 = k1/q1,k2/q2
    corrected_medians = factor1*median1-factor2*median2
    # Hệ số dựa trên shape đã biết trong DGP; không tuyên bố có SE chính xác từ dữ liệu thật.
    # Var(median) ~ 1/(4 n f(median_population)^2); scale được ước lượng bằng median/q.
    se1 = factor1*(median1/q1)/(2*np.sqrt(n1)*stats.gamma.pdf(q1,k1))
    se2 = factor2*(median2/q2)/(2*np.sqrt(n2)*stats.gamma.pdf(q2,k2))
    median_half = stats.norm.ppf(.975)*np.sqrt(se1*se1+se2*se2)
    return means,corrected_medians,welch_half,pooled_half,median_half,welch_df,welch_se,pooled_se


def simulation_study():
    # ADEMP: estimand là chênh lệch kỳ vọng mô hình, không phải hiệu ứng tổng thể đã biết.
    # Methods: mean difference; Gamma-corrected median difference with scenario shapes known.
    # Sampling CIs are Welch, pooled-t and the conditional Gamma-median Wald approximation.
    k1,s1,k2,s2 = gamma_parameters()
    pooled = np.r_[data()[0],data()[1]]
    reference_mean,reference_variance = pooled.mean(),pooled.var(ddof=1)
    common_k = reference_mean**2/reference_variance
    common_s = reference_variance/reference_mean
    scenarios = [
        ('Gamma_equal_distributions',common_k,common_s,common_k,common_s),
        ('Gamma_equal_means_diff_CV',k1,reference_mean/k1,k2,reference_mean/k2),
        ('Gamma_observed_means',k1,s1,k2,s2)]
    store('simulation_design',[dict(scenario=name,shape_T=a,scale_T=b,shape_C=c,scale_C=d,
        mean_T=a*b,mean_C=c*d,truth_delta=a*b-c*d,B=B_MAIN,
        sample_sizes='(7,9), (14,18), (28,36)',
        tests='H0: mean_T-mean_C = truth_delta; Type I, not power',
        permutation_in_study='not applied: equal means alone does not justify label exchangeability')
        for name,a,b,c,d in scenarios])
    metrics,coverage,differences = [], [], []
    for index,(scenario,a,b,c,d) in enumerate(scenarios):
        truth = a*b-c*d
        for n1,n2 in [(7,9),(14,18),(28,36)]:
            gen = rng(20,index,n1,0)
            treatment = gen.gamma(a,b,size=(B_MAIN,n1))
            control = gen.gamma(c,d,size=(B_MAIN,n2))
            mean,median,half_w,half_p,half_m,df_w,se_w,se_p = group_estimates(treatment,control,a,c)
            for method,values in [('Mean difference',mean),('Gamma median difference (shape known)',median)]:
                metrics.append(performance(scenario,n1,n2,method,values,truth))
            for name,values,half in [('Welch mean CI',mean,half_w),('Pooled-t mean CI',mean,half_p),
                                    ('Gamma median Wald CI (shape known)',median,half_m)]:
                row = coverage_row(scenario,n1,n2,name,values-half,values+half,truth)
                if name == 'Welch mean CI':
                    pvals = 2*stats.t.sf(abs((mean-truth)/se_w),df_w)
                    assert np.isclose(np.mean(pvals < ALPHA),row['rejection_under_true_H0'])
                elif name == 'Pooled-t mean CI':
                    pvals = 2*stats.t.sf(abs((mean-truth)/se_p),n1+n2-2)
                    assert np.isclose(np.mean(pvals < ALPHA),row['rejection_under_true_H0'])
                coverage.append(row)
            independent = rng(20,index,n1,1)
            independent_t = independent.gamma(a,b,size=(B_MAIN,n1))
            independent_c = independent.gamma(c,d,size=(B_MAIN,n2))
            independent_median = group_estimates(independent_t,independent_c,a,c)[1]
            differences.append(compare_losses(scenario,n1,n2,mean,median,independent_median,truth))
    m = store('simulation_bias_mse',metrics)
    c = store('simulation_coverage',coverage)
    store('simulation_method_comparison',differences)
    study_figures(m,c)


def power_study():
    # H0 cố định delta=0 ngày. Đối chứng giữ cùng mean; điều trị có mean=mean_C+effect.
    # Khi mean thay đổi và shape giữ cố định, variance Gamma cũng thay đổi theo mean.
    k1,_,k2,_ = gamma_parameters()
    pooled = np.r_[data()[0],data()[1]]
    control_mean = pooled.mean()
    common_k = control_mean**2/pooled.var(ddof=1)
    effects = [-20.,0.,10.,20.,30.,40.]
    families = [('Gamma_common_shape',common_k,common_k),('Gamma_observed_shapes',k1,k2)]
    store('power_design',[dict(scenario=name,H0_delta=0,control_mean=control_mean,
        shape_T=a,shape_C=c,effects_days=str(effects),B=B_MAIN,alpha=ALPHA,
        sample_sizes='(7,9), (14,18), (28,36)',alternative='two-sided',
        tests='Welch; pooled-t; Gamma-median Wald with known shapes; no permutation power',
        interpretation='effect=0: Type I control; effect != 0: power',
        limitation='fixed shapes are assumed; Gamma variances change with alternative means')
        for name,a,c in families])
    rows = []
    for family,(scenario,a,c) in enumerate(families):
        baseline_sd = np.sqrt((control_mean**2/a+control_mean**2/c)/2)
        for index,effect in enumerate(effects):
            treatment_mean = control_mean+effect
            assert treatment_mean > 0
            b,d = treatment_mean/a,control_mean/c
            for n1,n2 in [(7,9),(14,18),(28,36)]:
                gen = rng(30,family,index,n1)
                treatment = gen.gamma(a,b,size=(B_MAIN,n1))
                control_samples = gen.gamma(c,d,size=(B_MAIN,n2))
                mean,median,_,_,median_half,df_w,se_w,se_p = group_estimates(treatment,control_samples,a,c)
                p_welch = 2*stats.t.sf(abs(mean)/se_w,df_w)
                p_pooled = 2*stats.t.sf(abs(mean)/se_p,n1+n2-2)
                median_se = median_half/stats.norm.ppf(.975)
                p_median = 2*stats.norm.sf(abs(median)/median_se)
                for name,pvalues in [('Welch',p_welch),('Pooled t',p_pooled),('Gamma median Wald',p_median)]:
                    rows.append(power_row(scenario,name,n1,n2,pvalues,effect,0.,
                        effect_standardized=effect/baseline_sd,mean_T=treatment_mean,mean_C=control_mean,
                        shape_T=a,shape_C=c,variance_T=a*b*b,variance_C=c*d*d,unit='days'))
    power = store('simulation_power',[row for row in rows if row['quantity'] == 'power'])
    control = store('simulation_power_null_control',[row for row in rows if row['quantity'] == 'type_I_error'])
    plot_power_results(power,control,'truth','Alternative mean difference (days); fixed H0=0',finish_figure)


if __name__ == '__main__':
    main()
