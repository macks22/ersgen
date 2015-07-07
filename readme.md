A Generative Process for Traditional University Data

# Goals

1.  Characterize and visualize distributions of variables in ERS data.
2.  Define a generative process to produce a synthetic dataset of grades over a
    sequence of enrollment terms in a traditional university setting.
3.  Analyze the effect of tweaking various aspects of the process. Define
    different processes based on different models. The most effective models
    will be able to fit the generative process which most closely reproduces our
    data. Hence the best generative process defined will help lead to the most
    effective predictive model to capture the observed trends.
4.  Generate datasets of arbitrary sizes for experimentation and model checking
    purposes.

# Generative Processes

## Basic MLR Process

Let's begin with a model that has no knowledge of the actual data. It is purely
something I've come up with to experiment with the SCD-learned MLR model. Assume
our model is:

$$\hat{g_{s,c}} = b_s + b_c + p_s^T \cdot W \cdot f_{sc}$$

The naive generative process for this model will work as follows:

### Feature generation

1.  For each course:
    1.  Draw credit hours (chrs) from Poisson with rate 3.
    2.  Draw clevel $\in {1,2,3,4,5,6} \sim$
        Multinomial(0.4570, 0.1960, 0.2640, 0.0790, 0.0021, 0.0019).
    3.  Draw cumCGPA from normal centered at 3.0 with std of 0.5.
2.  For each student:
    1.  Draw alevel $\in {0,1,2,3,4} from
        Multinomial(0.31, 0.32, 0.21, .14, 0.02).
    2.  Draw hsgpa $\sim \mathbb{N}(3.0, 0.5)$, set initial cumGPA = hsgpa.
    3.  Draw gender $\sim bernoulli(0.5)$.
3.  For each term $t$:
    1.  For each student $s$:
        *   Choose number of courses $C_{t,s} \sim Poisson(4)$.
        *   Select $C_{t,s}$ courses in accordance with alevel. Go with same
            clevel as alevel with prob 0.7, with +- 1 with prob 0.2, and with +-
            2 with prob 0.1. Ensure chrs does not exceed 21.
        *   Determine grade in each course by:
            cumGPA * cumCGPA + Age/30 + Gender(0.2) + (alevel - 1)(clevel -1) -
            (chrs/3 - 1).
        *   Bound grades using logistic function.
    2.  Update cumGPA and cumCGPA to use for next term.

### MLR Parameter generation

Rather than using the ad-hoc approach described above, we can also define a
proper generative model.

1.  For each student $s$, draw $b_s \sim \mathbb{N}(\mu_s, \sigma_s)$.
2.  For each course $c$, draw $b_c \sim \mathbb{N}(\mu_c, \sigma_c)$.
3.  For each student $s$, draw $(m_s \in \mathbb{R}^l) \sim
    Dir(\boldsymbol{\alpha})$.
4.  For each model $l$, draw $(W_l \in \mathbb{R}^p) \sim MvN(\mu, \Sigma)$.
5.  Draw features ??

