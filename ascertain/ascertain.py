"""Statistics of `l = a + b` variables under ascertainment.

We assume that the individual expections are zero, the sum of the variances
of `a` and `b` are equal to one, and `a` and `b` are independent from each
other, under no ascertainment.
"""
import scipy.stats as st


def threshold(K):
    """Threshold above which the outcome is one.

    The variables are assumed to follow normal distributions.

    Parameters
    ----------
    K : Prevalence in the population.
    """
    return st.norm.isf(K)


def mean(v, K, P):
    """Expectation of the underlying variable under ascertainment.

    Parameters
    ----------
    v : Variance of the variable under no ascertainment.
    K : Prevalence in the population.
    P : Proportion of cases in the sampled population.
    """
    t = threshold(K)
    return v * st.norm.pdf(t) * (P - K) / (K * (1 - K))


def second_moment(v, K, P):
    """Second moment of the underlying variable under ascertainment.

    Parameters
    ----------
    v : Variance of the variable under no ascertainment.
    K : Prevalence in the population.
    P : Proportion of cases in the sampled population.
    """
    t = threshold(K)
    return v + v * v * st.norm.pdf(t) * t * (P - K) / (K * (1 - K))


def variance(v, K, P):
    """Variance of the underlying variable under ascertainment.

    Parameters
    ----------
    v : Variance of the variable under no ascertainment.
    K : Prevalence in the population.
    P : Proportion of cases in the sampled population.
    """
    return second_moment(v, K, P) - mean(v, K, P)**2


def product_expectation(va, vb, K, P):
    """Expectation of `ab` under ascertainment.

    Parameters
    ----------
    va : Variance of the first variable under no ascertainment.
    vb : Variance of the second variable under no ascertainment.
    K : Prevalence in the population.
    P : Proportion of cases in the sampled population.
    """
    t = threshold(K)
    return (P / K - (1 - P) / (1 - K)) * st.norm.pdf(t) * t * va * vb


def covariance(va, vb, K, P):
    """Variance of the underlying variable under ascertainment.

    Parameters
    ----------
    va : Variance of the first variable under no ascertainment.
    vb : Variance of the second variable under no ascertainment.
    K : Prevalence in the population.
    P : Proportion of cases in the sampled population.
    """
    ma = mean(va, K, P)
    mb = mean(vb, K, P)
    return product_expectation(va, vb, K, P) - ma * mb
