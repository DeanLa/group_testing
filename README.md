# Langsam Adaptive Bayesian Strategy for the Group Tetsting Problem
## Introduction

The $(N, q)$ combinatorial group testing problem is defined on set of $N$ items, either of which can be GOOD with probability $q$ or BAD with probability $1-q$. In order to achieve full calssification there exists a **Group Testing Machine** (GTM) that accepts any finite number of items and can produce one of two results:
1. The test **passes** if all items in the test are GOOD.
2. The test **fails** if one or more items are BAD.

A passed test achieves full classification for the set of tested items, where a failed test requires further investigation on the **contaminated** set.

The $(\infty, q)$ or $q$-endless group testing problem represent an endless (Or very large) line of items and a GTM, and its purpose is to find an optimal **strategy** of group testing that minimizes the amount ratio $\frac{tests}{item}$ (equivalently maximizes $\frac{items}{test}$). It is important to note that for $N \ge 8$, the optimal strategy unknown as this problem is super-exponential and becomes very large, very quickly, let alone when $N$ itself grows to inifinity. I start with discussing known strategies for the q-endless problem. I will then compare which strategy is better for different values of $q$. I will discuss the main conclusions from those strategies and use those conclusions as assumptions for the main problem which is presented next.

## Goal
The $(\infty, \bar{q})$ is a $q$-endless problem where $q$ in unknown to the tester, thus no strategy can be chosen in advance. $\bar{q}$ is assumed to be a consant value, thus pretty simple strategies can be chosen after an initial "warming" period.
the $(\infty, \tilde{q})$ is also a $q$-endless problem where $q$ in unknown to the tester, moreover, simulating real-life, $\tilde{q}$ can change with every batch of items (imagine each day in a factory, after shutting the production machines off and on, $\tilde{q}$ may change). For this problem, a more complex, Bayesian approach is proposed.