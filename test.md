
# Up and Running with Python
<!-- _class: lead -->

Nandan Rao
2020

---

## What we will cover...

1. What is Python?
2. Running Python.
3. Writing Python.

---

## What is Python?

Python refers to both:

1. A program (it runs on a computer).
2. A programming language.

The program's purpose is to **interpret** the language and convert it into instructions for a computer.

---


## What is Python?

<!-- _class: sidecode -->

```shell
> which python

> which python3
```

The program's purpose is to **interpret** the language and convert it into instructions for a computer:

1. A program (it runs on a computer).
2. A programming language.



---

# Classifiers for Text Mining
<!-- _class: lead -->


Now we'll go over some basic classifiers.

---

## Outline

* Emprical risk minimization and $p(y,x)$
* $p(x)$ when $X$ is language
* Generative vs discriminative classifiers
* Hyperparameters in BOW framework
* Towards supervising the embedded space

---

## Emprical Risk Minimization

Consider an input space $X$, a discrete output space $Y$ and a function, $g: X \rightarrow Y$ which predicts a value for $y$ given an input $x$. Consider also a _loss function_ $\ell: Y,Y \rightarrow \mathbb{R}$.

The _risk_ of the classifier $g$ is defined as:

$$
R(g) = \mathbb{E}_{X,Y} [ {\ell(g(x), y)} ]
$$
$$
R(g) = \sum_{y \in Y} \int_X \ell(g(x), y) \  p(x,y) \ dx
$$

---

## Emprical Risk Minimization

We estimate the risk with its finite-sample approximation, the _empirical risk_:

$$
R(g) = \frac{1}{n} \sum_{i=1}^n \ell(g(x_i), y_i)
$$

Note, that this is only an approximation of $\mathbb{E}_{X,Y}$ if all the pairs:

$$
x_i, y_i \in p(X,Y)
$$

---

## Emprical Risk Minimization

$$
x_i, y_i \in p(X,Y)
$$

Implies that $p(Y|X)$ and $p(X)$ are the same in our validation distributions as they are in our target distribution.

Ways to deal with changes in any of the component parts of the joint distribution are covered in the literature of _domain adaptation_.

---


## Naive Bayes
<!-- _class: sidemath -->
$$
p(y = 1 | x) = \frac{p(x,y=1)}{\sum_y p(x, y=y_i)} \\

p(y = 1 | x) = \sigma \bigg(  \log \frac{p(x,y_1)}{p(x,y_0)} \bigg) \\

p(y = 1 | x) = \sigma \bigg(  (\pi_1 -\pi_0)^Tx + \log \frac{p(y_1)}{p(y_0)} \bigg)
$$

Let's see how a multinomial naive Bayes can be related to logistic regression:

Where $\pi_i$ denotes the log probability parameters of the multinomial distribution representing $p(X|y_i)$

---

## Filter
<!-- _class: sidecode -->

```python
names = ['Foo', 'Bar', 'Baz']

def we_like(name):
    return name != 'Bar'

[n for n in names if we_like(n)]
```

Filter is used to remove certain elements from a list.

We can filter a list by adding an **if statement** to a list comprehension.

Like all if statements, the `if` keyword is followed by a boolean.

The element is included in the new list only if the boolean is true.

---


## Logistic Regression

Logistic regression models the posterior probability as:

$$
p(y|x; \beta) = \sigma(\beta^Tx)
$$

where the logistic function $\sigma$ is given by:

$$
\sigma(n) = \frac{1}{1 + \exp^{-n}}
$$

This can be fit via maximum likelihood ($\ell := -p(y|x; \beta)$) or by any other convex surragote of the 0-1 error ($\ell := 1\{ g(x) \neq y \}$)

---


## Naive Bayes

As mentioned, Naive Bayes predicts based on the posterior calculated from the modelled joint distribution:

$$
p(y|x) = \frac{p(x|y)p(y)}{p(x)}
$$

It's called "naive" because we make the (extreme) simplifying assumption that all the $p(x_i|y)$ are independent and thus $p(x|y) = \prod_i p(x_i|y)$.

---


## Running Python

<!-- _class: sidecode -->

```shell
> which python

> which python3
```

Python does not have a GUI, so we run it from a terminal.

Let's make sure we can find the program (the "executable").

`which` is a program, available in your terminal, that tells you where on your computer a given executable lives.

---

## Running Python

Let's run python:

```shell
> python3

Python 3.8.2 (default, Feb 26 2020, 22:21:03)
[GCC 9.2.1 20200130] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>>
```
---

## Writing Python

Python is a programming language.

It's named after Monty Python and was created by Guido van Rossum in 1991.

There are two major versions that are still circulating: 2 & 3.

Python2 has been dying a long, slow death for many years now. We will be using Python3 for this course.

---

## A python file

Create a new file in vscode and call it `hello.py`.

Write the following in the file:

```python
message = 'Hello CodeOp!'
print(message)
```

---

## A python file

In your terminal:

1. Navigate to the directory of the `hello.py` file you just created.
2. Run the file with python: `python3 hello.py`.


---

## VSCode and Python

1. Install the "Python" extension (by Microsoft)
2. Install some python libraries on your computer (pip, pylint, pytest, pytest-xdist)
