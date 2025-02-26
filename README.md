# 🚀 Step-by-step roadmap to becoming a Data Scientist

## ✨ by Timur Bikmukhametov

---

## 📌 Table of Contents
- [Motivation](#motivation)
- [The Goal of the Roadmap](#the-goal-of-the-roadmap)
- [READ THIS BEFORE YOU START](#read-this-before-you-start)
- [Disclaimers](#disclaimers)
- [Roadmap Overview](#roadmap-overview)
- [1. Python](#1-python)
  - [1.1 Introduction](#11-introduction)
  - [1.2 Data Manipulation](#12-data-manipulation)
  - [1.3 Data Visualization](#13-data-visualization)
    - [Intro](#intro)
    - [Deeper Dive](#deeper-dive)
  - [1.4 Selected Practical Topics](#14-selected-practical-topics)
    - [Topic 1: Python environments and how to set it up with Conda](#topic-1-python-environments-and-how-to-set-it-up-with-conda)
    - [Topic 2: Demystifying methods in Python](#topic-2-demystifying-methods-in-python)
    - [Topic 3: Python clean code tips and formatting](#topic-3-python-clean-code-tips-and-formatting)
    - [Topic 4: Python imports](#topic-4-python-imports)
    - [Topic 5: Python decorators](#topic-5-python-decorators)
- [2. Data Science / ML Introduction](#2-data-science--ml-introduction)
  - [2.1 Introduction](#21-introduction)
  - [2.2 Basic probability, statistics, and linear algebra](#22-basic-probability-statistics-and-linear-algebra)
    - [Linear algebra](#linear-algebra)
    - [Probability and Statistics](#probability-and-statistics)
  - [2.3 Supervised learning](#23-supervised-learning)
    - [Linear regression](#linear-regression)
    - [Logistic regression](#logistic-regression)
    - [Gradient boosting](#gradient-boosting)
    - [Random Forest](#random-forest)
    - [k Nearest Neighbours (k-NN)](#k-nearest-neighbours-k-nn)
  - [2.4 Unsupervised learning](#24-unsupervised-learning)
    - [Clustering](#clustering)
    - [Dimensionality reduction](#dimensionality-reduction)
- [3. Data Science / ML Deep Dive](#3-data-science--ml-deep-dive)
  - [3.1 Selected Practical Topics](#31-selected-practical-topics)
    - [Feature selection](#feature-selection)
    - [Feature importance](#feature-importance)
    - [Model metrics evaluation](#model-metrics-evaluation)
    - [Cross-validation](#cross-validation)
  - [3.2 Neural Networks Introduction](#32-neural-networks-introduction)
  - [3.3 Optimization with Python](#33-optimization-with-python)
    - [Introduction to mathematical optimization with Python](#introduction-to-mathematical-optimization-with-python)
    - [Bayesian Optimization](#bayesian-optimization)
    - [Optimization with SciPy](#optimization-with-scipy)
    - [Interactive playground of several optimization methods](#interactive-playground-of-several-optimization-methods)
    - [Additional resources](#additional-resources)
- [4. MLOps for Data Scientists](#4-mlops-for-data-scientists)
  - [4.1 Introduction](#41-introduction)
  - [4.2 Model registry and experiment tracking](#42-model-registry-and-experiment-tracking)
  - [4.3 ML Pipelines](#43-ml-pipelines)
  - [4.4 Model Monitoring](#44-model-monitoring)
  - [4.5 Docker basics](#45-docker-basics)
  - [4.6 Additional resources](#46-additional-resources)
- [5. Industrial AI Topics](#5-industrial-ai-topics)
  - [5.1 Signal processing](#51-signal-processing)
  - [5.2 Data-driven / Hybrid Process Modeling](#52-data-driven--hybrid-process-modeling)
    - [Process dynamics and control with video lectures](#process-dynamics-and-control-with-video-lectures)
    - [Hybrid modeling review](#hybrid-modeling-review)
    - [Data-driven modeling of dynamical systems](#data-driven-modeling-of-dynamical-systems)
    - [Physics-Informed Machine Learning](#physics-informed-machine-learning)
  - [5.3 Process Control and MPC](#53-process-control-and-mpc)
  - [5.4 Anomaly Detection](#54-anomaly-detection)
---

## Motivation
💡 Learning Data Science is both exciting and overwhelming. Years ago, there were limited resources; today, there's an ocean of materials. Where should you start? 🤔

🌍 Many aspire to solve real-world problems using AI in industrial sectors. Unfortunately, most learning materials don't focus on practical industry applications.

## The Goal of the Roadmap
✔️ This roadmap is your **step-by-step guide** to becoming a solid **Junior+/Middle Data Scientist** from scratch! 🚀

### Who is this roadmap for?
✅ Beginners looking for a structured learning path 📚
✅ Data Scientists preparing for job changes or promotions 💼
✅ Engineers transitioning into Data Science 🛠️
✅ Those interested in **Industrial AI** 🏭

## READ THIS BEFORE YOU START
❗ **You will never feel completely "ready."** There's always more to learn in Python, Machine Learning, and Optimization. This roadmap will help you **build strong fundamentals** 💪

✔️ **If you're a beginner**: Start with Python + ML Basics and build a small project 💻
✔️ **If you have experience**: Pick topics where you need improvement 🔍

## Disclaimers
⚠️ No affiliations with recommended courses—these are **handpicked based on experience** 🔍
💰 Many resources are free, but some paid options provide better structure 🏆
💡 Coursera offers **financial aid**—I used it as a student! 🎓

---

## Roadmap Overview
📌 **Learning is a journey, not a sprint!**
✅ Start with **Python + ML Basics** ➡️ Build a project ➡️ Progress to **Advanced Topics & MLOps**
✅ If aiming for **Industrial AI**, complete the core ML topics first.

---

## 1. Python
### 1.1 Introduction
Python is the most widely used programming language in Data Science. It’s powerful, easy to learn, and has a vast ecosystem of libraries for data analysis, visualization, and machine learning.

Life is too short, learn Python. Forget R or S or T or whatever other programming language letters you see. And for God’s sake, no Matlab in your life should exist.

💡 **Your goal?** Get comfortable with Python basics and then dive into data manipulation and visualization—essential skills for any Data Scientist!

🔹 **Paid Courses:**
- 🎓 [Basic Python - CodeAcademy](https://www.codecademy.com/learn/learn-python-3)
- 🎓 [Python Programming - DataCamp](https://app.datacamp.com/learn/skill-tracks/python-programming)

🔹 **Free Courses:**
- 🎓 [FutureCoder.io (Hands-on)](https://futurecoder.io/)
- 🎥 [Dave Gray's Python Course](https://www.youtube.com/watch?v=qwAFL1597eM)
- 🛠️ [Mini-projects - freeCodeCamp](https://www.youtube.com/watch?v=8ext9G7xspg)

---

### 1.2 Data Manipulation
Data manipulation is the **core skill** for a Data Scientist. You’ll need to clean, transform, and analyze data efficiently using **Pandas and NumPy**.

- 📊 [Kaggle Pandas Course](https://www.kaggle.com/learn/pandas)
- 📚 [MLCourse.ai - Data Manipulation](https://mlcourse.ai/book/topic01/topic01_intro.html)
- 🔢 [Numpy Basics](https://github.com/ageron/handson-ml2/blob/master/tools_numpy.ipynb)
- 🏋️ [Pandas Exercises](https://github.com/guipsamora/pandas_exercises)

---

### 1.3 Data Visualization
Data visualization helps **communicate insights** effectively. Learning how to use Matplotlib, Seaborn, and Plotly will allow you to create compelling charts and dashboards.

💡 **Your goal?** Understand different types of plots and when to use them.

#### Intro
- 📊 [MLCourse.ai - Data Visualization](https://mlcourse.ai/book/topic02/topic02_intro.html)

#### Deeper Dive
- 🎨 [Matplotlib Examples](https://matplotlib.org/stable/gallery/index.html)
- 📊 [Seaborn Examples](https://seaborn.pydata.org/examples/index.html)
- 📈 [Plotly Interactive Plots](https://plotly.com/python/)

---

### 1.4 Selected Practical Topics
Once you’re comfortable with Python, these **practical topics** will help you **write cleaner, more efficient code** and work effectively in real projects.
#### Topic 1: Python environments and how to set it up with Conda
- 🔗 [Guide to Conda Environments](https://whiteboxml.com/blog/the-definitive-guide-to-python-virtual-environments-with-conda)

#### Topic 2: Demystifying methods in Python
- 🧐 [Understanding Python Methods](https://realpython.com/instance-class-and-static-methods-demystified/)

#### Topic 3: Python clean code tips and formatting
- 🧼 [Clean Code Principles](https://github.com/zedr/clean-code-python)
- 📝 [PEP8 Formatting Guide](https://realpython.com/python-pep8/)
- 🛠️ [Using Black Formatter](https://www.python-engineer.com/posts/black-code-formatter/)
- 🔍 [Linting with Flake8 & Pylint](https://www.jumpingrivers.com/blog/python-linting-guide/)

#### Topic 4: Python imports
- 📦 [Understanding Python Imports](https://realpython.com/python-import/)

#### Topic 5: Python decorators
- 🎭 [Guide to Python Decorators](https://realpython.com/primer-on-python-decorators/)

---

---

## 2. Data Science / ML Introduction
### 2.1 Introduction
- 🎓 [Machine Learning Course - Andrew Ng](https://www.coursera.org/learn/machine-learning?specialization=machine-learning-introduction)
  - Covers regression and classification, core ML concepts.
  - Coursera offers financial aid if needed.

---

### 2.2 Basic Probability, Statistics, and Linear Algebra
#### Linear Algebra
- 🎥 [3Blue1Brown Linear Algebra Series](https://www.3blue1brown.com/topics/linear-algebra?ref=mrdbourke.com)
- 📖 [Python Linear Algebra Tutorial - Pablo Caceres](https://pabloinsente.github.io/intro-linear-algebra)

#### Probability and Statistics
- 📊 [Statistics Crash Course - Adriene Hill](https://www.youtube.com/playlist?list=PL8dPuuaLjXtNM_Y-bUAhblSAdWRnmBUcr)
- 📖 [Learn Statistics with Python - Ethan Weed](https://ethanweed.github.io/pythonbook/landingpage.html)

---

### 2.3 Supervised Learning
#### Linear Regression
- 📖 [Nando de Freitas Lectures - UBC](https://www.youtube.com/playlist?list=PL05umP7R6ij2XCvrRzLokX6EoHWaGA2cM)
- 🛠️ [Closed-form & Gradient Descent Implementation](https://sebastianraschka.com/Articles/2014_pca_step_by_step.html)

#### Logistic Regression
- 📖 [MLCourse.ai - Logistic Regression](https://mlcourse.ai/book/topic05/topic05_intro.html)
- 📊 [Odds Ratio Interpretation](https://mmuratarat.github.io/2019-09-05/odds-ratio-logistic-regression)

#### Gradient Boosting
- 📖 [MLCourse.ai - Gradient Boosting](https://mlcourse.ai/book/topic10/topic10_gradient_boosting.html)
- 📑 [XGBoost Original Paper](https://arxiv.org/pdf/1603.02754.pdf)
- 🎮 [Gradient Boosting Demo - Alex Rogozhnikov](https://arogozhnikov.github.io/2016/06/24/gradient_boosting_explained.html)

#### Random Forest
- 🎥 [Nando de Freitas Lectures](https://www.youtube.com/playlist?list=PL05umP7R6ij2XCvrRzLokX6EoHWaGA2cM)
- 📖 [Bagging & Random Forest - MLCourse.ai](https://mlcourse.ai/book/topic05/topic05_intro.html)

#### k Nearest Neighbours (k-NN)
- 📖 [Understanding k-NN](https://mmuratarat.github.io/2019-07-12/k-nn-from-scratch)

---

### 2.4 Unsupervised Learning
#### Clustering
- 📌 [k-Means from Scratch](https://mmuratarat.github.io/2019-07-23/kmeans_from_scratch)
- 📌 [DBScan Clustering](https://github.com/christianversloot/machine-learning-articles/blob/main/performing-dbscan-clustering-with-python-and-scikit-learn.md)

#### Dimensionality Reduction
- 📉 [PCA - Sebastian Raschka](https://sebastianraschka.com/Articles/2014_pca_step_by_step.html)
- 🎨 [t-SNE Visualization](https://distill.pub/2016/misread-tsne/)
- 📈 [UMAP Explanation](https://pair-code.github.io/understanding-umap/)

---

## 3. Data Science / ML Deep Dive
### 3.1 Selected Practical Topics
#### Feature selection
- 📖 [Comprehensive Guide on Feature Selection - Kaggle](https://www.kaggle.com/code/prashant111/comprehensive-guide-on-feature-selection/notebook#Table-of-Contents)

#### Feature importance
- 📖 [Interpretable ML Book - Linear Models](https://christophm.github.io/interpretable-ml-book/limo.html)
- 📖 [Interpretable ML Book - Logistic Models](https://christophm.github.io/interpretable-ml-book/logistic.html)
- 🎥 [Tree-based Feature Importance - Sebastian Raschka](https://www.youtube.com/watch?v=ycyCtxZ0a9w)
- 📖 [Permutation Feature Importance - Interpretable ML Book](https://christophm.github.io/interpretable-ml-book/feature-importance.html)
- 🛠️ [SHAP Library Documentation](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html)

#### Model metrics evaluation
- 📖 [Regression Metrics - H2O Blog](https://h2o.ai/blog/2019/regression-metrics-guide/)
- 📖 [Classification Metrics - Evidently AI](https://www.evidentlyai.com/classification-metrics)

#### Cross-validation
- 📖 [Cross-validation Guide - Neptune AI](https://neptune.ai/blog/cross-validation-in-machine-learning-how-to-do-it-right)

---

### 3.2 Neural Networks Introduction
- 🎓 [Deep Learning Specialization - Andrew Ng](https://www.coursera.org/specializations/deep-learning)

---

### 3.3 Optimization with Python
#### Introduction to mathematical optimization with Python
- 📖 [Numerical Optimization - Indrag49](https://indrag49.github.io/Numerical-Optimization/)

#### Bayesian Optimization
- 🎮 [Bayesian Optimization Playground - Distill.pub](https://distill.pub/2020/bayesian-optimization/)
- 📖 [Bayesian Optimization Theory - Nando de Freitas](http://haikufactory.com/files/bayopt.pdf)

#### Optimization with SciPy
- 📖 [SciPy Optimization Overview](https://caam37830.github.io/book/03_optimization/scipy_opt.html)
- 📖 [Optimization Constraints with SciPy - Towards Data Science](https://towardsdatascience.com/introduction-to-optimization-constraints-with-scipy-7abd44f6de25)
- 📖 [SciPy Optimization Tutorial](https://jiffyclub.github.io/scipy/tutorial/optimize.html#)
- 📖 [Optimization in Python - Duke University](https://people.duke.edu/~ccc14/sta-663-2017/14C_Optimization_In_Python.html)

#### Interactive playground of several optimization methods
- 🎮 [Optimization Playground - Ben Frederickson](https://www.benfrederickson.com/numerical-optimization/)

#### Additional resources
- 📖 [Numerical Optimization Book - Jorge Nocedal](https://www.amazon.ca/Numerical-Optimization-Jorge-Nocedal/dp/0387303030)
- 📚 [Awesome Optimization Resources](https://github.com/ebrahimpichka/awesome-optimization)

---

## 4. MLOps for Data Scientists
### 4.1 Introduction
- 🎓 [MLOps Zoomcamp - DataTalksClub](https://github.com/DataTalksClub/mlops-zoomcamp/tree/main/01-intro)

---

### 4.2 Model Registry and Experiment Tracking
- 📖 [Model Registry - Neptune AI](https://neptune.ai/blog/ml-model-registry)
- 📖 [Experiment Tracking - Neptune AI](https://neptune.ai/blog/ml-experiment-tracking)
- 🛠️ [Hands-on Example - DataTalksClub](https://github.com/DataTalksClub/mlops-zoomcamp/tree/main/02-experiment-tracking)

---

### 4.3 ML Pipelines
- 📖 [Building End-to-End ML Pipelines - Neptune AI](https://neptune.ai/blog/building-end-to-end-ml-pipeline)
- 📖 [Best ML Workflow and Pipeline Orchestration Tools - Neptune AI](https://neptune.ai/blog/best-workflow-and-pipeline-orchestration-tools)
- 🛠️ [ML Pipelines with Mage/Prefect - DataTalksClub](https://github.com/DataTalksClub/mlops-zoomcamp/tree/main/03-orchestration)

---

### 4.4 Model Monitoring
- 📖 [MLOps Monitoring Guides - Evidently AI](https://www.evidentlyai.com/mlops-guides)
- 🎓 [MLOps Zoomcamp - Model Monitoring](https://github.com/DataTalksClub/mlops-zoomcamp/tree/main/05-monitoring)

---

### 4.5 Docker Basics
- 🎥 [Docker Crash Course - Nana](https://www.youtube.com/watch?v=3c-iBn73dDE)

---

### 4.6 Additional Resources
- 📖 [MLOps Roadmap 2024 - Marvelous MLOps](https://marvelousmlops.substack.com/p/mlops-roadmap-2024)

---

---

## 5. Industrial AI Topics
### 5.1 Signal Processing
- 🎓 [Signal Processing Course - Mike Cohen (Paid)](https://www.udemy.com/course/signal-processing/?couponCode=2021PM20)
- 📖 [Fourier Transform & Filters](https://en.wikipedia.org/wiki/Fourier_transform)

---

### 5.2 Data-driven / Hybrid Process Modeling
#### Process Dynamics and Control with Video Lectures
- 🎥 [Process Dynamics and Control - Mun.ca](https://www.mun.ca/engineering/crise/about-us/our-people/process-dynamics-and-control/)

#### Hybrid Modeling Review
- 📖 [Hybrid Modeling Review - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2772508123000546)
- 📖 [Hybrid Modeling Research - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0306261918309218)

#### Data-driven Modeling of Dynamical Systems
- 🎥 [Steve Brunton’s Course on Dynamical Systems](https://www.youtube.com/watch?v=Kap3TZwAsv0&list=PLMrJAkhIeNNR6DzT17-MM1GHLkuYVjhyt)

#### Physics-Informed Machine Learning
- 🎥 [Brunton's Course on Physics-Informed ML](https://www.youtube.com/watch?v=JoFW2uSd3Uo&list=PLMrJAkhIeNNQ0BaKuBKY43k4xMo6NSbBa)
- 🛠️ [PySINDy Library](https://pysindy.readthedocs.io/en/latest/examples/2_introduction_to_sindy/example.html)

---

### 5.3 Process Control and MPC
- 🎓 [Process Control with Python - Hedengren](https://apmonitor.com/pdc/index.php/Main/CourseSchedule)
- 📖 [Practical Process Control - Opticontrols](https://blog.opticontrols.com/)
- 🎥 [MPC and MHE with Casadi](https://www.youtube.com/watch?v=RrnkPrcpyEA&list=PLK8squHT_Uzej3UCUHjtOtm5X7pMFSgAL)
- 🛠️ [HILO-MPC Library](https://github.com/hilo-mpc/hilo-mpc)
- 🛠️ [do-mpc Library](https://www.do-mpc.com/en/latest/)

---

### 5.4 Anomaly Detection
- 📖 [Anomaly Detection Methods Review - ACM](https://dl.acm.org/doi/abs/10.1145/1541880.1541882)
- 📖 [Anomaly Detection with Python - Neptune AI](https://neptune.ai/blog/anomaly-detection-in-time-series)
- 📖 [Deep Learning Anomaly Detection](https://arxiv.org/pdf/2211.05244)
- 🛠️ [Time Series Anomaly Detection Libraries](https://github.com/rob-med/awesome-TS-anomaly-detection)
