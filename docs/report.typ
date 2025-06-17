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
 * Assignment 3 - Numerical Linear Algebra*
])

#align(center, text(15pt)[
  Arthur Rabello Oliveira
  #footnote[#link("https://emap.fgv.br/")[Escola de Matemática Aplicada, Fundação Getúlio Vargas (FGV/EMAp)], email: #link("mailto:arthur.oliveira.1@fgv.edu.br")], Henrique Coelho Beltrão #footnote[#link("https://emap.fgv.br/")[Escola de Matemática Aplicada, Fundação Getúlio Vargas (FGV/EMAp)], email: #link("mailto:henrique.beltrao@fgv.edu.br")]

  #datetime.today().display("[day]/[month]/[year]")
])

#align(center)[
  *Abstract*\
  _coming soon_
]


//  TABLE OF CONTENTS ----------------------------------
#outline()
#pagebreak()

= Introduction
<section_introduction>

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

where $chi_m$ is the chi-squared distribution with $m$ degreees of freedom, better discussed in @section_chi_square_distribution.

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

With a fixed $m = 100$, when $n -> oo$ we can see the distribution approaching $N(0, 1)$, as shown in @section_inner_product_histograms


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
$
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
  image("images/max_corr_100.png", width: 80%),
) <plot_max_corr_100>

#figure(
  image("images/max_corr_500.png", width: 80%),
) <plot_max_corr_500>

#figure(
  image("images/max_corr_1000.png", width: 80%),
) <plot_max_corr_1000>

#figure(
  image("images/max_corr_10000.png", width: 80%),
) <plot_max_corr_10000>

#figure(
  image("images/max_corr_100000.png", width: 80%)
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
This theoretical approximation gives a value in the general vicinity of the observed peak (around 0.42). The discrepancy arises because the variables ${C_(i j)}$ are not perfectly independent (for instance, $C_(1,2)$ and $C_(1,3)$ both depend on column $A_1$) and their distribution is only approximately normal. Nonetheless, this formula correctly shows that the peak of the distribution is determined by the dimensions $m$ and $n$.

In conclusion, the observed distribution is a *Gumbel distribution*. This arises because we are plotting the maximum of a very large number of approximately independent, normally-distributed random variables (#link("https://en.wikipedia.org/wiki/Cosine_similarity")[the cosine similarities]).

= Complexity
<section_complexity>

= Another Maximum Distribution
<section_another_maximum_distribution>

= Conclusion
<section_conclusion>

#bibliography("bibliography.bib")