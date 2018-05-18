"""Statistics of `l = a + b` variables under ascertainment.

We assume that the individual expections are zero, the sum of the variances
of `a` and `b` are equal to one, and `a` and `b` are independent from each
other, under no ascertainment.
"""
from numpy import sqrt
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


def alpha(va, vb, K, P):
    v0 = variance(va, K, P)
    v1 = variance(vb, K, P)
    rho = covariance(va, vb, K, P)
    v = v0 + v1
    return 1 + sqrt((v - 2*rho)/(v + 2*rho))


def m_mean(va, vb, K, P):
    return (mean(va, K, P) + mean(vb, K, P)) / 2


def m_second_moment(va, vb, K, P):
    v0 = second_moment(va, K, P)
    v1 = second_moment(vb, K, P)
    v01 = product_expectation(va, vb, K, P)
    return (v0 + 2 * v01 + v1)/4


def corrected_mean(va, vb, K, P):
    a = alpha(va, vb, K, P)
    return mean(va, K, P) - a * m_mean(va, vb, K, P)


def corrected_second_moment(va, vb, K, P):
    a = alpha(va, vb, K, P)
    v0 = second_moment(va, K, P)
    v01 = product_expectation(va, vb, K, P)
    vm = m_second_moment(va, vb, K, P)
    return v0 - a * (v0 + v01) + a * a * vm
