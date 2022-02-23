# Problem Statement

We have a set of locations L which we want to take as our input space of size n where n is the number of locations considered and some budget B. We want to compute some allocation A denoting the number of wind turbines to be allocated to that location.
$$
L = {\{l_1,l_2,...,l_n\}}
$$

$$
\forall i \in \{1,..n\}:l_i = \{loadfactor_{mean},loadfactor_{variance}\}
$$

$$
A = \{a_1,a_2,...,a_n\}
$$

$$
\forall i \in \{1,..n\}:a_i\in \Z^+
\\
\sum a_i \leq B
$$

We want to define some function O such that we minimise a multi-objective function F(A,L) where our multiple objectives refer to reducing wasted generation capacity (1 - loadfactor) and targetting consistent output levels by reducing output variance. We can also include cost factors here such as connection costs and transport costs aiming to minimise those.
$$
O:L \rightarrow A
$$
