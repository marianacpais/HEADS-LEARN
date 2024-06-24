# Lesson 03 - Graphical Models - Bayesian Networks

## Summary

### Introduction

"Bayesian statistical methods allow taking into account prior knowledge when analyzing data, turning the data analysis a process of updating that prior knowledge with biomedical and health-care evidence" Lucas P. (2004)

"Bayesian networks offer a general and versatile approach to capturing and reasoning with uncertainty in medicine and health care." Lucas P. (2004)

“Each variable is a node in the network, and its dependency defined by their ascendants in the network, quantified by a conditional probability table.” Darwiche A. (2010)

- Consist of
  - Qualitative model (structure)
    - variables (nodes)
    - dependencies between variables (edges)
  - Quantitative model (parameters)
    - conditional probabilities for each variable state

### Qualitative Model

- Defined by experts
- Categorical variables are included, considering three groups:
  - Theories: risk or conditioning factors (e.g. smoking)
  - Hypotheses: study outcomes (e.g. SARS diagnosis)
  - Observations: symptoms or measurements (e.g. fever)
- Dependencies are defined by the expert, taking causality into account:
  - Exposition → Outcome (e.g. smoking “causes” heart attack)
  - Outcome → Symtom (e.g. SARS “causes” dyspnoea)

- Interpreting the qualitative model
  - When defined automatically from data, the quantitative model CANNOT be interpreted as a causal model!
  - Dependencies are interpreted as association only.
  - Direction (A → B) indicates that having information on the conditioning (A) gives more information on the conditioned (B) than the reverse.
  - Inexistence of a dependency is a clear assumption of independence.
  - But the model can be adjusted manualy after being created from the data! e.g. reversing an edge direction

### Quantitative Model

- Conditional probabilities at each node
- Defined by:
  - Expert (subjectively)
    - e.g. physician estimates that 1/3 of the patients is older than 65
  - Theories known in literature (e.g. meta-analysis)
    - e.g. prevalence of obstructive sleep apnea is 4% in men and 2% in women
  - Data from an original study
    - e.g. proportion of cases with dyspnoea among SARS patients

- Interpreting the quantitative model
  - When defined by an expert, it is based on a low level of evidence.
  - When defined by published theories, it is based on a high level of evidence.
  - When defined by an original study, it should be evaluated according to the study design:
    - Cohort: represents risks if the variable being conditioned was not controled (outcome included); otherwise, information interpretable as association.
    - Case-control: represents risks if the variable being conditioned was not controled (outcome excluded); otherwise, information interpretable as association.
    - Clinical trials: represents risks according to the randomization or information interpretable as association, for example, if sample stratification was made.

- Interpreting resulting probabilities
  - When no evidence is included in the network:
    - Probabilities are the marginal probabilities for each category, considering the dependencies of the multivariate model - a priori risk
  - When evidence is included in the network:
    - Probabilities are the marginal probabilites for the subgroups of patients defined by the evidence, considering the entire multivariate model – a posteriori risk

### Learning Structure from data

- **Search-and-score methods**:
  - search algorithm: select subset of (high-quality) BNs
  - quality measure (score): decide which one of the (candidate) networks is the best
  - **Defined from data (hill-climbing)**:
    1. Start with no dependencies
    2. Compute the likelihood of observing the data given the model
    3. Test all possible single dependencies, computing the respective likelihood of the data
    4. If the likelihood increases above a given threshold, insert that dependency in the model
    5. Go to 3, until threshold is not reached by any new dependency

- **Constraint-based structure learning**:
  - identifies ADG structure that best encodes a set of conditional dependence and independence assumptions

- **The classifier approach**
  - “I don't care if the OR for wine drinking and myocardial infaction is of risk (>1) or protection (<1), or even if it is 1.2 or 12; I care that I can use that info to discriminate between patients who will develop the disease and those who won't!”
  - While research often focus on finding (accurate) estimates of association between factors and outcomes (e.g. OR, RR, …), there is no certainty that a model thereby defined will act as a good classifier.
  - Classification is a basic task in data analysis and pattern recognition that requires the construction of a classifier, that is, a function that assigns a class label to instances described by a set of attributes.
  - **Naive Bayes classifier**
    - One of the most effective classifiers, in the sense that its predictive performance is competitive with state-of-theart classifiers, is the so-called naive Bayesian classifier.
    - This classifier learns from data the conditional probability of each attribute Fi given the outcome O (in similar approach as case-control studies study the odds of exposure given the outcome).
    - Classification is then done by applying the Bayes rule to compute P(O | f).
    - This computation makes a strong independence assumption:
      - all the attributes are conditionally independent given the value of the outcome. By independence we mean probabilistic independence, that is, A is independent of B given C whenever P(A | B , C) = P(A | C) for all possible values of A, B and C, whenever P(C) > 0
  - **Augmenting the Naive Bayes classifier**
    - Interdependencies between attributes can be addressed directly by allowing an attribute to depend on other non-class attributes.
    - However, techniques for learning unrestricted Bayesian networks oſten fail to deliver lower zero-one loss than naive Bayes!
      - One possible reason for this is that full Bayesian networks are oriented toward optimizing the likelihood of the training data rather than the conditional likelihood of the class attribute given a full set of other attributes.
      - Another possible reason is that full Bayesian networks have high variance due to the large number of parameters estimated.
      - An intermediate alternative technique is to use a less restrict structure than naive Bayes.
    - **Tree-Augmented Naive Bayes classifier**
      - Tree augmented naive Bayes is a semi-naive Bayesian Learning method.
      - It relaxes the naive Bayes attribute independence assumption by employing a tree structure, in which each attribute only depends on the class and one other attribute.
      - A maximum weighted spanning tree that maximizes the likelihood of the training data is used.
      - Chow and Liu (1968) proposed a method that efficiently constructs a maximum weighted spanning tree which maximizes the likelihood that the training data was generated from the tree.
      - The weight of an edge in the tree is the mutual information of the two attributes connected by the edge. TAN extends this method by using conditional mutual information as weights.
      - Due to the relaxed attribute independence assumption, TAN considerably reduces the bias of naive Bayes at the cost of an increase in variance.
      - Empirical results show that it substantially reduces zero-one loss of naive Bayes on many data sets and that of all data sets examined it achieves lower zero-one loss than naive Bayes more oſten than not.
