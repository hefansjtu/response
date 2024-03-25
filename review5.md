**Questions:** 
-  From Figure 1, it seems that the method overfits the support vectors (the regression line touches all support vector outputs) and is almost piecewise linear in between. This is, according to me, a flaw of the method since smoothing does not lie in the choice of the kernel but only in the number of support vectors. If so, why not using linear spline approximation? Besides, with Gaussian kernels, overfitting is typical of excessively large bandwidths. This is consistent with Algorithm 1 trying to achieve an empirical error less than B: for small bandwidths, this may be impossible, even considering all points as support vectors. This suggest that Algorithm 1 tries to learn excessively large bandwidths/overfit well chosen points. In this context, the main merit of the proposed approach is to select iteratively support vectors. Yet, it can be argued that this greedy procedure highly depends on its initialization.

-   Looking at Figure 1 (a), it seems that the observations are very noisy in the first half (low frequency) and almost without noise in the second half (high frequency). If this is true, the interpretation and the need for local adaptivity of kernels may change. Can the authors explain how the the data has been generated?
## Figure 1: Linear Spline Approximation or Kernel-based Function Approximation?

We appreciate your insightful question regarding Figure 1, where our algorithm appears similar to linear spline interpolation. However, the primary reason for this phenomenon is **the utilization of an excessive amount of support data** in this particular toy experiment. (This approach was chosen to improve the performance of the traditional RBF kernel.) Consequently, our function converges to a nearly piece-wise linear function due to the abundance of support data. In this scenario, such a function performs exceptionally well, particularly for the high-frequency components.

Here we present another result from this experiment, this time with reduced support data. You can view the result in [Figure 1](https://github.com/icml2024-4062/Response/blob/main/toyexample-box.pdf). The highlighted red box shows the most challenging part: without a support data in the two peak, it is almost impossible for traditional kernel method with only single RBF kernel to fit. While benefited by the trainable bandwidths, our approach are much more flexible, could easily approximate these two peaks, which obviously cannot be achieved by linear spline.

Furthermore, the [original code](https://github.com/icml2024-4062/Response_Figure1/tree/main) for generating this figure is provided. The sampled data are generated as follows:
$$y = \sin(2x^3), \;\; n\sim \mathcal{U}[-0.25,0.25], \;\;\hat{y} = y+n.$$
For details on the code implementation, please refer to the 'data_reg.py' file. It's worth noting that the noise level remains consistent throughout the entire signal. The high-frequency component may appear to have smaller noise simply because the signal undergoes significant changes in this region, causing the noise-affected data points to resemble their neighbors more closely.

##

**Questions:** -   The requirement of empirical error less than B (Equation (10)) is the cornerstone of the method (and of the proposed theoretical guarantee) while I argue that it may not be achievable for all B and all bandwidths. Can the authors comment on this point and on the relevance of the approach when this level of error cannot be achieved?


## Overfitting or not: trade-off via $B$
We appreciate your insightful question regarding $B$. We completely agree that not every $B$ value can be achieved with any bandwidth. **We believe there exists a causal relationship between them. Specifically, we first determine the value of $B$, and then suitable bandwidths are obtained through optimization.** This approach aligns with most existing machine learning methods, where we first specify hyper-parameters and subsequently determine the model weights.

As discussed in Section 5, $B$ in our approach acts as a trade-off hyper-parameter. In many cases, our approach performs effectively when an appropriate strategy is employed to select $B$. Let's delve into this in more detail:

 - **The specific value of $B$ is contingent upon the noise-level.** Similar to most learning methods, smaller approximation errors ($B$) don't necessarily equate to better performance. It's natural to opt for larger $B$ values when prior knowledge indicates the presence of substantial noise, and vice versa. Otherwise, overfitting or underfitting may occur.

- **Moreover, the requisite number of support data to achieve $B$ hinges on the complexity of the data pattern,** or in other words, the sufficiency of sampling. If the data exhibits a simple pattern, attaining the desired $B$ is straightforward. Conversely, in cases of complex data, more support data is needed to achieve a desired level of error.

In practice, we employ a validation dataset to evaluate the noise level, which assists us in selecting an appropriate value for $B$. Typically, we strive to minimize the number of support data to improve the generalization ability of the final solution. Therefore, we adopt a greedy-like strategy in this regard. Furthermore, as discussed earlier, the number of support data is determined once the error level $B$ is fixed. Hence, in our experiments, we solely report the number of support data for the sake of clarity and ease of comparison.

##
**Questions:** 
-   Algorithm 1 is a greedy procedure, that seems ad hoc regarding the aim of learning the kernel mentioned in Equation (1). It has to be noted that Proposition 5.2 makes a theoretical link between Algorithm 1 and Problem (12). Nevertheless, this results looks artificial, first because it is about the training dataset, second because it is actually true for any admissible function f_z (and not only an optimal solution to (12)).
-   Overall, the equivalence of the problem with integral spaces of RKHSs (as stated in the conclusion) is not clear enough according to me.

## Why we construct model (12) and how we guarantee their equivalence?

Indeed, Equation (12) is specifically tailored for Algorithm 1 to facilitate understanding of its learning behavior. Our ultimate objective is to analyze Algorithm 1 from an approximation perspective, akin to previous studies [1]. To achieve this goal, we need to delineate the hypothesis space and the regularized learning model.

Currently, there is limited research on the analysis of asymmetric kernels, with only a few works concentrating on reproducible kernel Banach space [2-4]. However, proving the reproducible property of LAB RBF kernels poses a challenge, and it may not be valid.

Therefore, we opt for an alternative approach: leveraging the unique structure of LAB RBF kernels, we interpret our algorithm within an integral space of RKHSs. **This space is integral because the domain of bandwidths is continuous**, unlike in previous studies where it was discrete. By examining its sparsity (which we discuss in detail in another answer), we construct model (12). This model not only aids in explaining the origin of the proposed algorithm's performance but also lays a solid foundation for our future learning analysis.

Through this process, Proposition 5.2 serves as a formal statement linking our algorithm to the $\ell_0$-regularized learning model, demonstrating that our algorithm provides a $B$-optimal solution. **The inequality in Proposition 5.2 is essentially the definition of a $B$-optimal solution.** Thanks to your feedback, we have refined it to:
$$0\leq \frac{1}{N}\sum_{i=1}^N(f_{\mathcal{Z}_{sv},\Theta}(x_i)-y_i)^2 - \frac{1}{N}\sum_{i=1}^N(f_{z}(x_i)-y_i)^2 \leq B$$,
which is an immediate inference of Equation (10), with the optimal condition indicated by the first inequality.

Based on our ongoing analysis, we find that the optimal solution of model (12) demonstrates robust convergence performance due to the sufficiently large hypothesis space and strong $\ell_0$-norm regularization. However, this also makes direct optimization challenging. Our approach effectively addresses this challenge, providing an efficient and effective solution to this model.

[1] F. Cucker and D. X. Zhou. Learning Theory: An Approximation Theory Viewpoint.  Cam-  
bridge University Press Cambridge, 2007.
[2] Haizhang Zhang, Yuesheng Xu, and Jun Zhang. Reproducing kernel banach spaces for machine
learning. Journal of Machine Learning Research, 10(12), 2009.
[3] Lin, Rong Rong, Hai Zhang Zhang, and Jun Zhang. "On reproducing kernel Banach spaces: Generic definitions and unified framework of constructions." _Acta Mathematica Sinica, English Series_ 38.8, 2022.
[4] Altabaa, Awni, and John Lafferty. "Approximation of relation functions and attention mechanisms." _arXiv preprint arXiv:2402.08856_ ,2024.

##

**Questions:** -   The theory mentions integral space RKHS while in practice, a direct sum of RKHSs (such as for Multiple Kernel Learning) is used.
## The hypothesis is integral space rather than sum space
It's crucial to acknowledge that while we successfully confine the optimal solution within a sum space, this sum space undergoes **continuous changes** throughout the training process as the bandwidth dynamically adjusts (Please refer to Figure 2).  

Mathematically, let's assume a sum space of RKHSs generated from a bandwidth set $\Theta=\{\theta_1,\cdots, \theta_N\}\subset \Omega$ is denoted as $\mathcal{H}_\Theta$. In current works such as [], a pre-given $\Theta$ is utilized as a kernel dictionary. However, in our approach, $\Theta$ is obtained through optimization. Consequently, the hypothesis space involved in our approach is the union of all possible $\mathcal{H}_\Theta$, i.e.
$$\mathrm{Hypothesis\; Space:} \bigcup_{\Theta\subset\Omega}\mathcal{H}_\Theta = \mathcal{H}_\Omega.$$
Consequently, the hypothesis space remains an integral space rather than a fixed sum space.

This constitutes the most significant distinction between our work and existing approaches that rely on a fixed sum space of RKHSs as the hypothesis space. This disparity is further exemplified by the definition domain of kernels. In existing discussions of the sum space of RKHSs, the domain of kernels is discrete (typically a dictionary of kernels). However, in our paper, the domain of bandwidth is continuous, resulting in a continuous domain of kernels and a significantly larger hypothesis space than previously considered.

##
**Questions:** -   In Section 3.2, first, it is difficult to disentangle the contribution (the new model) from the literature (in particular regarding asymmetric kernel learning). 
## The contribution of model (4)/(7)
Our original motivation for proposing model (4)/(7) was to provide an answer to the question of **why we can apply asymmetric kernels in the classical Kernel Ridge Regression (KRR) formulation** $\alpha = (K+\lambda I)^{-1}Y$ from the perspective of asymmetric inner product.

**Contribution to Asymmetric Kernel-Based Learning**. Recent advancements in asymmetric kernel-based SVD [1] and SVM [2], in conjunction with our work, are all situated within the framework of LS-SVM [3]. However, our contribution lies in being the first to delineate the optimization process for asymmetric kernel-based regression from both primal (model (4)) and dual (model (7)) perspectives. In order to maintain compatibility with symmetric cases and ensure that the solution retains the form $\alpha = (K+\lambda I)^{-1}Y$, we meticulously design the optimization problem. This design is not a trivial extension of previous problems but a carefully crafted approach tailored to the specific characteristics of asymmetric kernels.

Moreover, this analytical expression serves as a foundation for our asymmetric kernel learning algorithm. Equipped with kernel learning algorithm, our algorithm exhibits excellent performance across general datasets, while surpassing previous methods particularly on datasets with asymmetric relationships. This highlights the significance of our contribution in advancing the field of asymmetric kernel-based learning.

**Contribution to Asymmetric Kernel Learning**. Previous works [4-5] directly apply asymmetric kernels or adjacency matrices in the formulation $\alpha = (K+\lambda I)^{-1}Y$. However, due to the lack of corresponding explanation, these works primarily focus on Gaussian processes where asymmetric kernels are considered as asymmetric correlation functions rather than asymmetric inner products.

Based on our study of model (4)/(7), we draw connections between them and the symmetric KRR model. Through this comprehensive analysis, we discover that there are actually two decision functions: $\alpha = (K+\lambda I)^{-1}Y$ and $\beta = (K^\top+\lambda I)^{-1}Y$, which do not exist in symmetric cases and have been overlooked by other works. Although in our subsequent approach we only utilize the former one, we discuss their relationship and the potential utilization of the latter may be studied in the future.

[1] Suykens, J. A. K. SVD revisited: A new variational principle, compatible feature maps and nonlinear extensions. Applied and Computational Harmonic Analysis, 40(3): 600–609, 2016.
[2] He, M., He, F., Shi, L., Huang, X., and Suykens, J. A. K.  Learning with asymmetric kernels: Least squares and  feature interpretation.  IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(8):10044–10054,  2023.
[3] Suykens, J. A. K. and Vandewalle, J. Least squares support vector machine classifiers.  Neural processing letters, 9: 293–300, 1999.
[4] Pintea, Silvia L., Jan C. van Gemert, and Arnold WM Smeulders. "Asymmetric kernel in Gaussian Processes for learning target variance." _Pattern Recognition Letters_ 108 (2018): 70-77.
[5] AlBahar, Areej, Inyoung Kim, and Xiaowei Yue. "A robust asymmetric kernel function for Bayesian optimization, with application to image defect detection in manufacturing systems." _IEEE Transactions on Automation Science and Engineering_ 19.4 (2021): 3222-3233.

##
 **Questions:** - Second, since all results are expressed with \phi  and  \psy, application with kernel from Equation (1) is not explicit.
## What is the $\phi$ and $\psi$ of LAB RBF kernels?
From Mercer's theorem we know that for traditional RBF kernels, there exists a feature mapping function $\phi_\sigma(\cdot):\mathbb{R}^d\to \mathcal{F}$ satisfying that  $\mathcal{K}_\sigma({\bf t},{\bf x}) = \langle \phi_\sigma({\bf t}), \phi_\sigma({\bf x})\rangle$.
Recall the definition of the proposed LAB RBF kernel function over dataset $\mathcal{X}=\{{\bf x}_1,\cdots,{\bf x}_N\}$ and corresponding bandwidths ${\bf\Theta}=\{\theta_1,\cdots,\theta_N\}$ in Equation (1), we can define
    $\phi({\bf t}) = [\phi^\top_{\theta_1}({\bf t})\;\;\phi^\top_{\theta_2}({\bf t})\;\; \cdots\;\;\phi^\top_{\theta_N}({\bf t})\;]^\top$,
    $\psi({\bf x}) = [\phi^\top_{\theta_1}({\bf x})\delta({\bf x}-{\bf x}_1)\;\; \phi^\top_{\theta_2}({\bf x})\delta({\bf x}-{\bf x}_2)\; \;\cdots\;\;\phi^\top_{\theta_N}({\bf x})\delta({\bf x}-{\bf x}_N)\;]^\top,$

where $\theta_i$ is the corresponding bandwidth for data ${\bf x}_i$, and $\delta(\cdot)$ is the Dirac delta function.
Then we can decompose the asymmetric LAB RBF kernels defined over a dataset $\mathcal{X}$ as the inner product of $\phi$ and $\psi$. That is, 
$$\mathcal{K}_{\bf\Theta}({\bf t},{\bf x}_i)= \langle \phi({\bf t}), \psi({\bf x}_i)\rangle= \exp\left\{- \|\theta_i\odot({\bf t}-{\bf x}_i)\|_2^2\right\},\qquad \forall {\bf t}\in\mathbb{R}^d, {\bf x}_i\in\mathcal{X}.$$



**Questions:**  -   In Section 5, the authors makes the link with Multiple Kernel Learning. However, MKL learns a kernel through a sum of kernels, not a single one. So how is it linked to the kernel in Equation (1), except assuming that the sum is 1-sparse?
## Link to Multiple Kernel Learning
Recalling the definition of LAB RBF kernels in Equation (1): $\mathcal{K}_\Theta(t,x_i) = \exp\left\{- \|\theta_i\odot(t-x_i)\|_2^2\right\}, \forall x_i\in\{x_1,\cdots,x_N\}$, where  **$N$ kernels with bandwidths $\{\theta_1,\cdots,\theta_N\}$ are involved**. Hence, the function estimated by LAB RBF kernels shares similarities with that of MKL to some extent.

Assume a kernel dictionary $\{\mathcal{K}_1,\;\cdots, \;\mathcal{K}_L\}$, and data points $\{x_1,\cdots,x_N\}$, the original MKL [1] considers a new kernel $\mathcal{K}=\sum_{l=1}^L\mu_l\mathcal{K}_l$, with the function $f(\cdot) = \sum_{n=1}^N a_n \mathcal{K}(\cdot, x_n)$. In this approach, only one RKHS (generated by $\mathcal{K}=\sum_{l=1}^L\mu_l\mathcal{K}_l$) is involved once the coefficient $\mu_l$ is fixed.

In a more general case of MKL, function are generated in the sum space of the $N$ RKHSs, formulated as follows.
$$f_{MKL}(\cdot) =\sum_{l=1}^L f_l(\cdot) = \sum_{l=1}^L  \sum_{n=1}^Ma_{n,l} \mathcal{K}_l(\cdot, x_n).$$
It is worth noting that in this approach, $a$ has $L\times N$ coefficients, and some work has proven that the number of non-zero coefficients of $a$ is less than the data number $N$ (please refer to Equation (1.15) in [2]).

Recalling Equation (1) and (8), our regressor is:
$$f_{LAB}(\cdot) = \sum_{n=1}^{N_{sv}} \alpha_{n} \mathcal{K}_n(\cdot, x_n) = \sum_{l=1}^L  \sum_{n=1}^{N_{sv}} \hat{a}_{l,n} \mathcal{K}_l(\cdot, x_n).$$
where $L={N_{sv}}$ kernels are involved (Here we use $\mathcal{K}_l$ instead of $\mathcal{K}_{\theta_l}$ for convenience in comparison). Thus, we could express $f_{LAB}$ in a similar form as $f_{MKL}$, and it is evident that 
$$\hat{a}_{l,n}=\begin{cases}
    \alpha_n,\;&\mathrm{if}\;n=m,\\
    0,\;&\mathrm{otherwise}.
    \end{cases}$$
Compared with results in2],  the sparsity in our approach is further enhanced ($N_{sv}<<N\times L$) because **we are able to train bandwidths rather than using manually designed kernel dictionary**. 

[1] Gönen, M. and Alpaydın, E. Multiple kernel learning algorithms.  The Journal of Machine Learning Research, 12:  2211–2268, 2011.
[2] Aziznejad, Shayan, and Michael Unser. "Multikernel regression with sparsity constraint." _SIAM Journal on Mathematics of Data Science_ 3.1: 201-224, 2021.

##
**Questions:**  
 1.   The paragraph "Enhanced generalization from the sparse regularization" states that "Proposition 5.2 indicates that functions derived from LAB RBF kernels exhibit sparse coefficients within  H_Ω". I do not understand (so I may disagree).
 2.  The same paragraph tells about an "implicit  ℓ0-related term" but the sparsity in support vectors looks totally artificial from Algorithm 1. This statement should be made clear.
## Sparsity of LAB RBF kernels is not artificial
In our approach, two levels of sparsity are observed:
 1. **Reduced Support Data**: The number of support data points is significantly lower than the total number of training data points. While this sparsity is artificially determined algorithmically, its **essential reason lies in the sufficiently large hypothesis space**. This expansive hypothesis space enables us to employ fewer support data points to effectively approximate the entire training dataset.
 2.  **Inherent Sparsity of LAB RBF Kernels**: The functions generated by LAB RBF kernels demonstrate sparsity within the hypothesis space (it is a sum space when the bandwidth $\Theta^*$ is determined), contributing to a more efficient representation of the data.
 Given support dataset $\{{\bf x}_i,y_i\}_{i=1}^{N_{sv}}$, the hypothesis space considered in this paper is defined as the linear span of the set $\{\mathcal{K}_\sigma(\cdot, {\bf x}_i)\}$, $\forall i=1,\cdots,{N_{sv}},\;\forall \sigma\in\Omega$. This space forms a subspace of$\mathcal{H}_\Omega$, and the function within it takes the formulation:
$$f(\cdot) = \int_\sigma f_\sigma(\cdot)d\mu(\sigma) = \int_\sigma \sum_{i=1}^{N_{sv}}\alpha_{\sigma,i}\mathcal{K}_\sigma(\cdot,{\bf x}_i)d\mu(\sigma),$$
In LAB RBF kernels, $N_{sv}$ bandwidths are optimized, therefore we constrain ${\mu}(\sigma) = \sum_{i=1}^{N_{sv}} \delta(\sigma - \theta_i)$. While our algorithm restricts the number of valid kernel functions to be finite, their bandwidths are adaptable through optimization. Consequently, the associated function space varies within the entire integral space $\mathcal{H}_\Omega$.
Then the function formulation becomes 
$$f(\cdot) = \sum_{\sigma\in\Theta} f_\sigma(\cdot) = \sum_{\sigma\in\Theta} \sum_{i=1}^{N_{sv}}\alpha_{\sigma,i}\mathcal{K}_\sigma(\cdot,{\bf x}_i),$$
where the size of $\alpha_{\sigma,i}$ is $|\Theta|\times {N_{sv}}$. **Without sparsity, $\mathcal{R}_0(f)$ approximately equals $|\Theta|\times {N_{sv}}$.**
The function estimated by LAB RBF kernels, generated from the same kernels and data, takes a formulation like:
$$f_{LAB}(\cdot) = \sum_{i=1}^{N_{sv}} \hat{\alpha}_{i} \mathcal{K}_{\theta_i}(\cdot, x_i) = \sum_{\sigma\in\Theta} \sum_{i=1}^{N_{sv}} \alpha_{\sigma, i} \mathcal{K}_\sigma(\cdot, x_i).$$
We say this function exhibits sparsity because 
$$\alpha_{\sigma, i}=\begin{cases}
    \hat{\alpha}_i,\;&\mathrm{if}\;\sigma=\theta_i,\\
    0.\;&\mathrm{otherwise}.
    \end{cases}$$
**This results in $\mathcal{R}_0(f_{LAB}) = N_{sv}$**. In this regard, $f_{LAB}$ demonstrates enhanced sparsity compared to a typical function within the final sum space $\mathcal{H}_{\Theta^*}$, not to mention functions within the integral space $\mathcal{H}_\Omega$. Additionally, we have prepared a figure to provide a clearer illustration [Coefficient Matrix](https://github.com/icml2024-4062/other-figures/blob/main/coefficient%20matrix.pdf).

##

**Questions:** -   In Section 3.3, there is no discussion regarding convexity (or not) of the optimization problems and the possibility to apply the KKT theory. Moreover, the end of this section is aimed at enlightening the method but is quite confusing: how changing a regularization term and flipping a sign (Line 218) can make the problem similar to something else? What do the authors mean by "a computation facilitated through the kernel trick", "as they exhibit distinct approximation errors", "the signs […] tend to be dissimilar"?

## Issues with Section 3.3

- The optimization problem discussed in Section 3.3 is a bilinear problem and thus non-convex. In this case, we obtain stationary points from the KKT conditions, which we refer to as 'KKT points' before. Thanks to your suggestions, we have make it more clear in the revised version.

- The discussion at the end of Section 3.3 aims to provide a broad overview of the Least Squares Support Vector Machine (LS-SVM) framework, illustrating that for different problem settings, there exist several variants, including our model. We have modified the description to enhance clarity.

- The sentences "a computation facilitated through the kernel trick", "as they exhibit distinct approximation errors", "the signs [...] tend to be dissimilar" signify the following: 

	In our asymmetric kernel-based model, there are two regressors $f_1(t) = \phi(t)^\top w^*$, and $f_2(t) = \psi(t)^\top v^*$. Theorem 3.3 indicates that their approximation errors can be computed using the kernel trick, specifically as ${\bf e}^*=\lambda(K(X,X)+\lambda I_N)^{-1}Y$ (where $e_i$ denotes the approximation error of $f_1$ on data $x_i$), and ${\bf r}^*= \lambda(K^\top(X,X)+\lambda I_N)^{-1}Y$ (where $r_i$ denotes the approximation error of $f_2$ on data $x_i$). Due to the asymmetry of $K(X,X)$, we generally observe $e_i\neq r_i$, indicating that $f_1(x_i)\neq f_2(x_i)$, and thus **the two regressors are not the same function**. Furthermore, from the objective function where $\sum_i e_i r_i$ is minimized. Consequently, we can infer that **$e_i$ and $r_i$ tend to have different signs, resulting in $e_i r_i <0$.** 
##
**Questions:**  
-   There seems to be something missing in the end of Equation (9).
-   Line 8 of Algorithm 1 is a gradient ascent step. Is it correct?
## Other minor issues
We have carefully checked the Equation (9), and we believe nothing is missing.
We have corrected this wrong expression of the Line 8 in Algorithm in the revised version, and it should be a gradient descent step. Thanks for your careful reading.


