#import "@preview/ctheorems:1.1.3": *
#import "@preview/plotst:0.2.0": *
#import "@preview/codly:1.2.0": *
#import "@preview/codly-languages:0.1.1": *
#codly(languages: codly-languages)

#show: codly-init.with()
#show: thmrules.with(qed-symbol: $square$)
#show link: underline
#show ref: underline

#set heading(numbering: "1.1.")
#set page(numbering: "1")
#set heading(numbering: "1.")
#set math.equation(
  numbering: "(1)",
  supplement: none,
)

#set par(first-line-indent: 1.5em,justify: true)
#show ref: it => {
  // provide custom reference for equations
  if it.element != none and it.element.func() == math.equation {
    // optional: wrap inside link, so whole label is linked
    link(it.target)[eq.~(#it)]
  } else {
    it
  }
}

#let theorem = thmbox("theorem", "Theorem", fill: rgb("#ffeeee")) //theorem color
#let corollary = thmplain(
  "corollary",
  "Corollary",
  base: "theorem",
  titlefmt: strong
)
#let definition = thmbox("definition", "Definition", inset: (x: 1.2em, top: 1em))
#let example = thmplain("example", "Example").with(numbering: "1.")
#let proof = thmproof("proof", "Proof")

//shortcuts

#let inv(arg, power) = $arg^(-power)$
#let herm(arg) = $arg^*$
#let transpose(arg) = $arg^T$
#let inner(var1, var2) = $angle.l var1, var2 angle.r$
#let Var(arg) = $"Var"(arg)$
#let int = $integral$

#align(center, text(20pt)[
 * Assignment 4 - Numerical Linear Algebra*
])

#align(center, text(15pt)[
  Arthur Rabello Oliveira
  #footnote[#link("https://emap.fgv.br/")[Escola de Matemática Aplicada, Fundação Getúlio Vargas (FGV/EMAp)], email: #link("mailto:arthur.oliveira.1@fgv.edu.br")], Henrique Coelho Beltrão #footnote[#link("https://emap.fgv.br/")[Escola de Matemática Aplicada, Fundação Getúlio Vargas (FGV/EMAp)], email: #link("mailto:henrique.beltrao@fgv.edu.br")]

  #datetime.today().display("[day]/[month]/[year]")
])

#align(center)[
  *Abstract*\

We performed a computational study of $m times n$ real Gaussian matrices whose entries are i.i.d.\ $N(0,1)$. First, for increasing values of the ambient dimension $m$ (with $n = 1000$) we sampled the $2$-norm of each column and verified that the empirical histograms are well-approximated by the $chi_m$ distribution, converging to $sqrt(m)$.  
Second, fixing $m = 100$ and letting $n$ vary from $10$ to $1000$, we examined all off-diagonal inner products; they converge to $N(0, m)$, confirming the classical central-limit prediction. Third, we estimated the *worst* non-orthogonality—the maximum absolute cosine similarity among every matrix's columns—over up to $K = 10^5$ independent realisations and showed that the resulting maxima follow a Gumbel law, in line with extreme-value theory. Finally, we derived and validated the algorithmic cost $O(m n^2)$ of this pipeline and quantified that $K approx 10^3$ already yields stable statistics.

]


//  TABLE OF CONTENTS ----------------------------------
#outline()
#pagebreak()

= Introduction
<section_introduction>

== From multivariate statistics to nuclear physics
<section_from_multivariate_statistics_to_nuclear_physics>

The systematic study of Gaussian random matrices #link("https://en.wikipedia.org/wiki/Wishart_distribution")[began with *John Wishart* ($1928$)], whose sample-covariance analysis produced the distribution that still underpins multivariate testing and Bayesian inference. Two decades later, #link("https://en.wikipedia.org/wiki/Wigner_semicircle_distribution")[*Eugene Wigner* introduced random matrices to model the energy levels of heavy nuclei], proving the celebrated semicircle law for their eigenvalue distribution. These twin origins—statistics and physics—sparked what is now called *Random Matrix Theory (RMT)*, whose cornerstones include the #link("https://arxiv.org/pdf/1003.2990")[Marchenko-Pastur law] for singular values of rectangular Gaussian matrices and sharp bounds on their extremes.

== Modern applications
<section_modern_applications>

Random-matrix ideas now inform topics as diverse as numerical conditioning, wireless communication, portfolio theory and, most recently, *deep learning*. Which is why we study Gaussian matrices in this assignment, focusing on their non-orthogonality and the distribution of their inner products.

= Norm Distribution (a)
<section_norm_distribution>

== The Chi-Square Distribution
<section_chi_square_distribution>

Here we construct a theoretical basis for our analysis of the histograms shown in @section_histograms 

When we generate a matrix $A in RR^(m times n)$, with $A_( i j) ~ N(0, 1)$ independent, each column $c_i$ is a gaussian vector in $RR^m$. if

$
  x = vec(X_1, X_2, dots.v, X_m) in RR^m
$

Is a column, then:

$
  V = norm(x)_2 = sqrt(sum_(i=1)^m X_i^2)\
  
  V^2 = sum_(i=1)^m X_i^2
$

Is of our interest. The expected value and variance are:

$
  EE[V^2] = EE[sum_(i=1)^m X_i^2] = sum_(i=1)^m EE[X_i^2] = m\

  Var(V^2) = Var(sum_(i=1)^m X_i^2) = sum_(i=1)^m Var(X_i^2) = 2m
$

But we know that if $X_i ~ N(0, 1)$ are independent:

$
  sum_(i = 1)^m X_i^2 ~ chi_m^2
$ <equation_normal_sum_is_chi>

where $chi_m$ is the chi-squared distribution with $m$ degrees of freedom, better discussed in @section_chi_square_distribution.

Taking the square root on @equation_normal_sum_is_chi, we have:

$
  V = norm(x)_2 = sqrt(sum_(i=1)^m X_i^2) ~ sqrt(chi_m^2) ~ chi_m
$

The 2-norm of a vector $x$ is distributed as a chi distribution with $m$ degrees of freedom, in order to understand the distribution for many values of $m$, we can calculate the expected value and variance of this distribution as a function of $m$. The PDF of the chi distribution (with $m$ degrees of freedom) is:

$
  f_V (phi) = 1 / (2^(m / 2 - 1) dot Gamma(m / 2))  phi^(m - 1) e^(-phi^2 / 2)
$ <PDF_chi>


So from #link("https://proofwiki.org/wiki/Expectation_of_Chi_Distribution")[this], the expected value is:

$
  EE(V) = sqrt(2) dot Gamma((m + 1) / 2) / Gamma(m / 2)
$ <expectation_chi>

And from #link("https://proofwiki.org/wiki/Variance_of_Chi_Distribution")[this], the variance:

$
  Var(V) =
  m - (sqrt(2) dot Gamma((m + 1) / 2) / Gamma(m / 2))^2
$ <variance_chi>

The #link("https://en.wikipedia.org/wiki/Stirling%27s_approximation#Stirling's_formula_for_the_gamma_function")[Stirling Approximation] provides a good approximation for the expected value and variance:

$
  EE(V) approx sqrt(m) dot (1 - 1 / (4 m) + O(1 / m^2))
$

$
  Var(V) approx 1 / 2 + O(1 / m)
$

== Histograms
<section_histograms>

#link("https://github.com/arthurabello/nla-assignment-4/blob/main/src/assignment.ipynb")[The first cell of this notebook] has as expected output, with input being matrices with fixed $n = 1000$ and $m in {10, 20, 100, 200, 1000, 2000}$, the following plots:


#figure(
  image("images/histo_10_1000.png", width: 80%),
  caption: [
    $10 times 1000$ gaussian matrix
  ]
) <histo_10_1000>

#figure(
  image("images/histo_20_1000.png", width: 80%),
  caption: [
    $20 times 1000$ gaussian matrix
  ]
) <histo_20_1000>

#figure(
  image("images/histo_100_1000.png", width: 80%),
  caption: [
    $100 times 1000$ gaussian matrix
  ]
) <histo_100_1000>

#figure(
  image("images/histo_200_1000.png", width: 80%),
  caption: [
    $200 times 1000$ gaussian matrix
  ]
) <histo_200_1000>

#figure(
  image("images/histo_1000_1000.png", width: 80%),
  caption: [
    $1000 times 1000$ gaussian matrix
  ]
) <histo_1000_1000>

#figure(
  image("images/histo_2000_1000.png", width: 80%),
  caption: [
    $2000 times 1000$ gaussian matrix
  ]
) <histo_2000_1000>

#table(
  columns: (auto, auto, auto, auto, auto),
  align: horizon,
  inset: 6pt,
  table.header(
    [$m$],
    [*approximate $mu_m$ (theory)*],
    [*$[mu ± 3sigma]$ (theory)*],
    [*observed spike*],
    [*visual range*],
  ),

  [10],   [$3.08$],  [$1.0 - 5.18$],  [$approx 3.1$],          [$1.2 -5.1$],
  [20],   [$4.42$],  [$2.3 - 6.52$],  [$approx 4.3 - 4.4$],      [$2.9 - 6.4$],
  [100],  [$9.98$],  [$7.9 - 12.1$],  [$approx 9.9 - 10.0$],     [$7.8 - 12.0$],
  [200],  [$14.12$], [$12.0 - 16.2$], [$approx 14.1$],         [$12.2 - 16.2$],
  [1000], [$31.61$], [$29.5 - 33.7$], [$approx 31.7 - 32.0$],   [$29.9 - 33.3$],
  [2000], [$44.72$], [$42.6 - 46.8$], [$approx 44.5 - 45.0$],   [$42.8 - 46.6$],
)

This table illustrates the expected value $mu_m$ and the range $[mu - 3sigma, mu + 3sigma]$ for @histo_10_1000 to @histo_2000_1000.

So apparently as $m$ grows, the size of the gaussian vectors rapidly converge to $sqrt(m)$, with small errors.

= Inner Products (b)
<section_inner_products>

Here we construct a theoretical basis for our analysis of the inner products shown in @section_inner_product_histograms.

When we generate a matrix $A in RR^(m times n)$, with $A_( i j) ~ N(0, 1)$ independent, each column $c_i$ is a gaussian vector in $RR^m$. If
The inner product of two gaussian vectors $x = (X_1, dots, X_n), y = (Y_1, dots, Y_n)$ is:

$
  Z = inner(x, y) = sum_(i = 1)^m X_i Y_i 
$

With $X, Y ~ N(0, 1)$. Since $X_i, Y_j$ are independent, we have:

$
  EE[Z] = sum_(i = 1)^m EE[X_i Y_i] = sum_(i = 1)^m EE[X_i] EE[Y_i] = 0\
  Var(Z) = sum_(i = 1)^m Var(X_i Y_i) = sum_(i = 1)^m EE[X_i^2] EE[Y_i^2] = sum_(i = 1)^m 1 = m
$

If $W = X_i Y_i$, we have:

$
  M_W (phi) = EE[e^(phi W)] = 1 / sqrt(1 - phi^2), abs(phi) < 1
$

Over all $W_i = X_i Y_i$:

$
  M_Z (phi) = EE[e^(phi Z)] = (M_W (phi))^m = (1 / sqrt(1 - phi^2))^m = (1 - phi^2)^(-m / 2), abs(phi) < 1
$

And magically:

$
  M_(Z / sqrt(m)) (phi) = (1 - phi^2 / m)^(-m / 2) => lim_(m -> oo) M_(Z / sqrt(m)) (phi) = e^(phi^2 / 2)
$

Precisely the moment generating function of a standard normal distribution, so as $m -> oo$:

$
  Z / sqrt(m) ~ N(0, 1)
$

And finally:

$
  Z ~ N(0, m)
$ <final_inner_product_distribution>

With a fixed $m = 100$, when $n -> oo$ we can see the distribution approaching $N(0, m)$, as shown in @section_inner_product_histograms

== Histograms
<section_inner_product_histograms>

The following plots are an expected output for the second cell of #link("https://github.com/arthurabello/nla-assignment-4/blob/main/src/assignment.ipynb")[this notebook], with input $m = 100, n in {10, 20, 30, 40, 50, 60, dots, 1000}$:

#figure(
  image("images/inner_100_10.png", width: 100%),
) <plot_inner_100_10>

#figure(
  image("images/inner_100_50.png", width: 100%),
) <plot_inner_100_50>


#figure(
  image("images/inner_100_100.png", width: 100%),
) <plot_inner_100_100>


#figure(
  image("images/inner_100_200.png", width: 100%),
) <plot_inner_100_200>

#figure(
  image("images/inner_100_400.png", width: 100%),
) <plot_inner_100_400>

#figure(
  image("images/inner_100_500.png", width: 100%),
) <plot_inner_100_500>

#figure(
  image("images/inner_100_700.png", width: 100%),
) <plot_inner_100_700>

#figure(
  image("images/inner_100_1000.png", width: 100%),
) <plot_inner_100_1000>


@plot_inner_100_10 $->$ @plot_inner_100_1000 shows that the distribution indeed approaches $N(0, 1)$

= The Maximum Distribution (c)
<section_maximum_distribution>

In this section, we analyze the distribution of the maximum non-orthogonality between columns of a Gaussian matrix. This non-orthogonality is quantified by the maximum absolute value of the cosine similarity between any two distinct column vectors. Specifically, for a matrix $A in RR^(m times n)$, we study the distribution of the random variable:

$
  M = max_(i != j) (|inner(A_i, A_j)|) / (norm(A_i) norm(A_j))
$ <eq_max_corr>

Our experiment generates $K$ independent realizations of this value, $M_1, M_2, ..., M_K$, by creating $K$ different Gaussian matrices of size $m=100, n=300$. The histograms later shown in @section_maximum_distribution_histograms display the empirical probability density function of this collection of maxima.

== Theoretical Framework and the Gumbel Distribution
<section_maximum_distribution_theory>

Let $C_(i j) = inner(A_i, A_j) / (norm(A_i) norm(A_j))$. For a given matrix $A$, we are examining the maximum of $N = n(n-1) / 2$ random variables, ${|C_(i j)|}_(1 <= i < j <= n)$. For $m=100$ and $n=300$, this is the maximum of $N = 44850$ values.

We are interested in the maximum of ${|C_(i j)|}$.
As established in previous sections:
- From part (a) (@section_norm_distribution), for large $m$, $norm(A_i)$ concentrates around $sqrt(m)$.
- From part (b) (@section_inner_products), $Z_(i j) = inner(A_i, A_j)$ is approximately $N(0, m)$.

Let's first characterize the distribution of a single variable $C_(i j)$.
$
  C_(i j) = Z_(i j) / (norm(A_i) norm(A_j)) approx (N(0, m)) / (sqrt(m) dot sqrt(m)) = (N(0, m)) / m
$
If a random variable $X ~ N(0, sigma^2)$, then $X/c ~ N(0, sigma^2/c^2)$. Thus:
$
  C_(i j) approx N(0, m / m^2) = N(0, 1/m)
$ <equation_normal_property>
So, the individual correlation values are approximately drawn from a normal distribution with mean 0 and a small variance of $1/m$.

Our analysis, however, concerns the variable $M = max_(i!=j) |C_(i j)|$. The parent distribution is therefore not $N(0, 1/m)$, but rather its absolute value, $|N(0, 1/m)|$. This is known as a #link("https://en.wikipedia.org/wiki/Folded_normal_distribution")[*folded normal distribution*].

The tail of the folded normal distribution behaves identically to the tail of the underlying normal distribution. According to #link("https://en.wikipedia.org/wiki/Extreme_value_theory")[*Extreme Value Theory*], the limiting distribution for the maximum of many i.i.d. variables from a parent distribution with an exponential tail (like the normal distribution) is the #link("https://en.wikipedia.org/wiki/Gumbel_distribution")[*Gumbel distribution*.]

The probability density function (PDF) for the Gumbel distribution is given by:
$
  f(x; mu, beta) = 1/beta e^(-(z + e^(-z)))\
  z = (x - mu) / beta
$
where $mu$ is the mode of the distribution (location parameter) and $beta$ is the scale parameter (proportional to the standard deviation).

== Analysis of the Histograms
<section_maximum_distribution_histograms>

#figure(
  image("images/max_corr_k100_m100_x_n300.png", width: 80%),
) <plot_max_corr_100>

#figure(
  image("images/max_corr_k500_m100_x_n300.png", width: 80%),
) <plot_max_corr_500>

#figure(
  image("images/max_corr_k1000_m100_x_n300.png", width: 80%),
) <plot_max_corr_1000>

#figure(
  image("images/max_corr_k10000_m100_x_n300.png", width: 80%),
) <plot_max_corr_10000>

#figure(
  image("images/max_corr_k100000_m100_x_n300.png", width: 80%)
) <plot_max_corr_100000>

The histograms generated, especially for large $K$ (e.g., $K=10000$ and $K=100000$ as shown in @plot_max_corr_10000 and @plot_max_corr_100000, respectively), exhibit the distinct features of a Gumbel distribution:
- A single peak (unimodal).
- Asymmetry with a more extended tail on the right side.

As we can observe, growing $K$ _(number of trials)_ leads to a smoother plot and a clearer shape of the distribution, which aligns with the theoretical expectations of the Gumbel distribution.

The observed mode of the distribution is around 0.42, which is consistent with theoretical predictions. The location parameter $mu$ can be approximated by:
$
  mu approx sqrt((2 ln(N)) / m) = sqrt((2 ln(n(n-1)/2)) / m)
$
This formula arises from the well-known approximation for the expected maximum of $N$ standard normal variables ($sqrt(2 ln N)$), applied to our standardized variables ${\|sqrt(m)C_(i j)\|}$.

For $m=100$ and $n=300$, we have $N=44850$:
$
  mu approx sqrt((2 ln(44850)) / 100) approx sqrt((2 dot 10.71) / 100) = sqrt(0.214) approx 0.462
$
This theoretical approximation gives a value in the general vicinity of the observed peak (around 0.42). The discrepancy arises, and will be more evident when discussing convergence at @section_complexity_convergence, because the variables ${C_(i j)}$ are not perfectly independent (for instance, $C_(1,2)$ and $C_(1,3)$ both depend on column $A_1$) and their distribution is only approximately normal. Nonetheless, this formula correctly shows that the peak of the distribution is determined by the dimensions $m$ and $n$.

In conclusion, the observed distribution is a *Gumbel distribution*. This arises because we are plotting the maximum of a very large number of approximately independent, normally-distributed random variables (#link("https://en.wikipedia.org/wiki/Cosine_similarity")[the cosine similarities]).

= Complexity
<section_complexity>

== Algorithm Complexity and Runtime
<section_complexity_runtime>

The complexity of the algorithm is determined by the main operations within each of the $K$ iterations.

The process begins by generating a Gaussian matrix of size $m times n$, which has a time complexity of $O(m n)$. We then calculate the L2-norm for $n$ columns of length $m$ using `norms = np.linalg.norm(A, axis=0)`, an operation with $O(m n)$ complexity. The most computationally expensive step is the calculation of the Gram Matrix via `G = A.T @ A`. This matrix multiplication of an $n$ x $m$ matrix with an $m$ x $n$ matrix has a complexity of $O(m n^2)$. Subsequent operations, including the outer product ($O(n^2)$), element-wise division ($O(n^2)$), and maximum extraction ($O(n^2)$), are less expensive.

The total complexity for a single iteration is the sum of these steps, dominated by the Gram matrix calculation:
$
  &O(text("One Iteration")) = O(m n) + O(m n) + O(m n^2) + O(n^2) = O(m n^2)
$
Therefore, for $K$ iterations, the total complexity of our algorithm is $O(K m n^2)$. This implies that the runtime should scale linearly with $K$ and $m$, and quadratically with $n$. We can verify this empirically.

#figure(
  image("images/complexity_analysis.png", width: 100%),
  caption: [
    Runtime for varying $K$, $n$, and $m$
  ]
)<plot_complexity_analysis>

As predicted, @plot_complexity_analysis confirms our theoretical model. The plots show that the runtime scales linearly with $K$ and $m$, and quadratically with $n$. This empirically verifies the algorithm's overall complexity.

== Algorithm Convergence and Choosing an Appropriate K
<section_complexity_convergence>

The question _"What value of $K$ is good for a good estimate of the expected maximum?"_ is about statistical convergence. $K$ represents our sample size, which must be large enough for our statistics (like the mean and the histogram's shape) to be stable and reliable.

We first examine this convergence for the baseline case of $(m=100, n=300)$.
To compute these maxima over many iterations (up to $K=10^5$), we used Multiprocessing. A simple way to visualize convergence is to plot the running average of the maximum correlation as $K$ increases. We expect this average to fluctuate for small $K$ and converge to a stable value as $K$ grows.

#figure(
  image("images/complexity_mean_convergence_parallel_k100000_m100_n300.png", width: 100%),
  caption: [
    Convergence of @eq_max_corr as $K$ grows to $10^5$ with $(m,n) = (100,300)$
  ]
)<plot_complexity_mean_convergence_100_300>

From @plot_complexity_mean_convergence_100_300, we can observe that, for this specific case of $(m=100, n=300)$, the mean of the maximum correlation converges to approximately $0.42$ as $K$ increases. The convergence can be summarized as follows:
- For $K < 100$, the estimate is very noisy and unreliable.
- For $100 <= K < 1000$, the estimate begins to stabilize, despite some minor yet visible fluctuations.
- For $K >= 1000$, our estimate becomes very stable and converges smoothly to $approx 0.42$.

A $K$ value in the range of $10^3$ to $10^4$ is therefore a good choice for this specific problem. However, the rate of convergence for other pairs $(m,n)$ may differ. The convergence speed depends on the variance of the underlying Gumbel distribution, which is proportional to the square of its scale parameter, $beta$. This parameter can be derived from established results in Extreme Value Theory.

The derivation begins with the known asymptotic scale parameter for the maximum of $N$ standard normal variables, $beta_N approx 1/sqrt(2 ln N)$. In our case, the standardized variables are $Y_(i j) = sqrt(m)C_(i j) ~ N(0, 1)$. Since our variable of interest, $M = max|C_(i j)|$, is scaled by a factor of $1/sqrt(m)$ relative to the maximum of the standardized variables, the scale parameter is also scaled by this factor:
$
  beta_(m, n) = beta_N / sqrt(m) approx (1 / sqrt(2 ln N)) / sqrt(m) = 1 / sqrt(2 m ln(N))
$
This result implies that the variance is inversely proportional to $m ln(n)$. This leads to the prediction that distributions for larger $(m,n)$ pairs, having smaller variance, should converge more rapidly.

#figure(
  image("images/complexity_mean_convergence_parallel_k10000_m500_n1500.png", width: 100%),
  caption: [
    Convergence of @eq_max_corr as $K$ grows to $10^4$ with $(m,n) = (500,1500)$
  ]
)<plot_complexity_mean_convergence_500_1500>

In @plot_complexity_mean_convergence_500_1500, we observe a similar convergence pattern for $(m=500, n=1500)$, but with a more rapid stabilization of the mean. The fluctuations are smaller, and the mean converges to $approx 0.22$.

There are two paths we can take to ensure convergence:
=== Method 1: Adaptive K
<method_adaptive_k>
One could design an algorithm that stops once the running mean has stabilized. For instance, it could monitor the standard deviation of the running average over a recent window of trials and halt when this stability metric falls below a set tolerance.

- *Pros:* This approach is computationally efficient, performing only the necessary number of trials for each case. A clear advantage is that it adapts to the specific convergence behavior of each $(m,n)$ pair, ensuring each pair has a "good enough" $K$ with a clean stopping criterion.
- *Cons:* It introduces new arbitrary parameters (the window size and the tolerance threshold, and justifying those can become just as arbitrary as justifying a fixed $K$) and its sequential nature makes it difficult to parallelize efficiently.

=== Method 2: Fixed K
<method_fixed_k>
The second approach is to select a single, fixed, and sufficient value of $K$ for all experiments.

- *Pros:* This method is simple to implement, easily parallelizable (as the number of tasks is known beforehand), and ensures all results are directly comparable under identical sampling conditions. By choosing a $K$ that is sufficient for our "noisiet" case, we can guarantee that all other cases are also more than sufficiently sampled. There is no doubt about the statistical stability of any of your results.
- *Cons:* It is less computationally efficient, as it may "over-sample" cases that converge quickly.

The main goal of this project is to analyze and report on the statistical properties of random matrices, not to develop or showcase the most efficient simulation algorithm., hence why, for this project, we chose Method 2. The key is to select a $K$ based on a "worst-case" or slower-converging scenario. Since the smaller $(m,n)$ pairs are more variable, a $K$ value that is sufficient for them will be more than adequate for all larger, more stable pairs. A choice of $K=2500$ was therefore selected as a robust value that guarantees all our experiments are well-converged and our comparisons are statistically sound. It also produces results that are easier to interpret and compare across different $(m,n)$ pairs. This value of $K$ is sufficient to ensure convergence for the more variable cases (like $(m=100, n=300) #text("and") (m=100, n=100)$) and is more than adequate for all other pairs $(m,n)$ we tested.

This justifies our choice of a fixed $K=2500$ for the final experiments, as this value is sufficient for the more variable cases and thus more than adequate for all others. Nonetheless, the method of adaptive $K$ is a valid alternative and was also implemented in the notebook, allowing users to explore convergence behavior interactively. Here is the distribution of the maximum correlation for $(m,n) = (100,100)$, using the adaptive method with a window size of 25 and a tolerance of $10^(-5)$:
#figure(
  image("images/max_corr_k2135_m100_x_n100.png", width: 80%),
  caption: [
    Distribution of @eq_max_corr for $(m,n) = (100,100)$. $#text("window")=25 #text("tolerance")=10^(-5)$
  ]
)<plot_max_corr_adaptive_100_100>

For this case, the adaptive method stopped after $K=2135$ iterations. All other pairs $(m,n)$ should therefore converge with a similar or smaller $K$.


= Analysis of Maximum Correlation for Varying Dimensions
<section_another_maximum_distribution>

Once more with:

$
  M_(m, n) = max_(i != j) abs(inner(A_i, A_j)) / (norm(A_i) norm(A_j)), A in RR^(m times n), A_(i j) ~ N(0, 1)
$

Having established that the maximum correlation follows a Gumbel distribution @section_maximum_distribution and determined a robust methodology for sampling it @section_complexity_convergence, we now broaden our investigation. This section analyzes how the parameters of the Gumbel distribution change for various matrix dimensions, revealing the interplay between the ambient dimension, $m$, and the number of vectors, $n$.

For each pair $(m, n)$, we generated $K = 2500$ i.i.d. realizations of $M_(m, n)$. As established in @section_complexity_convergence, this fixed number of trials ensures that our results are statistically robust and directly comparable across all configurations. However, if you are interested in seeing the results of the adaptive method at @method_adaptive_k, they can be found at the second to last cell of #link("https://github.com/arthurabello/nla-assignment-4/blob/main/src/assignment.ipynb")[the notebook]. The parallel method was used for these computations, and a Gumbel distribution was fitted to each resulting histogram:

#grid(
  columns: (1fr, 1fr),
  gutter: auto,
  [
    #image("images/max_corr_k2500_m100_x_n100.png", width: 100%)
  ],
  [
    #image("images/max_corr_k2500_m100_x_n300.png", width: 100%)
  ]
)

#grid(
  columns: (1fr, 1fr),
  gutter: auto,
  [
    #image("images/max_corr_k2500_m200_x_n200.png", width: 100%)
  ],
  [
    #image("images/max_corr_k2500_m200_x_n600.png", width: 100%)
  ]
)

#grid(
  columns: (1fr, 1fr),
  gutter: auto,
  [
    #image("images/max_corr_k2500_m500_x_n500.png", width: 100%)
  ],
  [
    #image("images/max_corr_k2500_m500_x_n1500.png", width: 100%)
  ]
)

#grid(
  columns: (1fr, 1fr),
  gutter: auto,
  [
    #image("images/max_corr_k2500_m1000_x_n1000.png", width: 100%)
  ],
  [
    #image("images/max_corr_k2500_m1000_x_n3000.png", width: 100%)
  ]
)

We know from @equation_normal_property that the distribution of $C_(i j) = inner(A_i, A_j) / (norm(A_i) norm(A_j))$ is approximately $N(0, 1/m)$. So taking the absolute value gives a folded normal distribution that decays like $exp((-m phi^2) / 2)$.

The maximum of $N = (n (n - 1)) / 2$ converges to a Gumbel law, as discussed in @section_maximum_distribution_theory. Hence every histogram has the same shape, a sharp mode with a  long tail, regardless of $(m, n)$. Only the 2 Gumbel parameters change:

$
  mu_(m, n) = sqrt((2 ln(N)) / m)\

  beta_(m, n) = 1 / sqrt(2 m log N)
$ <equation_gumbel_parameters>

where $N = (n (n - 1)) / 2$ is the number of distinct pairs $(i, j)$.

The derivation of $beta$ is analogous to the one for $mu$.

One could note that the mode $mu$ moves as $m$ or $n$ are fixed. To better understand this. Set $N = (n (n - 1)) / 2$ and $sigma^2 = 1 / m$. With only the dominant terms in @equation_gumbel_parameters:

$
  mu_(m, n) approx sigma sqrt(2 log N) = sqrt((2 log N) / m)
$ <equation_rule_mu_growth>

These equations explain the trends visible in the plots. The mode $mu$ is influenced by two competing factors: it increases very slowly with the number of vectors ($n$, as $sqrt(ln n)$), but decreases more significantly with the dimension of the space ($m$, as $1/sqrt(m)$). This tells us that increasing dimensionality has a stronger effect on reducing the maximum correlation than increasing the number of vectors has on raising it.

= Conclusion
<section_conclusion>

This report systematically investigated the geometric properties of random Gaussian matrices, confirming that their non-orthogonality is governed by a competition between the number of vectors ($n$) and the dimension of the space ($m$). We showed that the L2-norms of columns concentrate sharply around $sqrt(m)$ (Chi distribution), and that inner products between columns are approximately Normal, reflecting near-orthogonality in high dimensions. The maximum correlation, quantifying the greatest non-orthogonality, follows a Gumbel extreme value law, with parameters determined by $(m, n)$. 

+ *Increasing the number of vectors* ($n$) for a fixed dimension ($m$) increases the expected maximum correlation.

+ *Increasing the dimension of the space* ($m$) decreases the expected maximum correlation, even when the number of vectors $n$ increases proportionally. That is, it has stronger effect in reducing the maximum correlation than increasing $n$ has in raising it.

Thus, the "blessing of dimensionality" ensures that large random matrices, despite their randomness, exhibit highly predictable and quantifiable structure.