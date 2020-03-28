# why

_An exploration of the world of interpretable machine learning__

Machine learning has become ubiquitous. Media coverage on ML's effectiveness proliferate, both in professional networks and popular media while ML experts continue to be in high demand. Online courses place education on popular ML methods at anyone's fingertips - from the basics to the cutting edge. And tools like scikit-learn or high-level interfaces to TensorFlow and PyTorch, make training a supervised model as easy as calling `.fit(X, y)`.

Fortunately, there are many real-life problems that can be solved by doing what these methods are great at – learning complex functions that map from an input to an output space. These machine learning applications create enormous value today and many quickly become engineering rather than data science problems.

On the other hand, there is a vast array of business questions that cannot be easily solved using the standard ML toolbox: Why did the model reject this applicant? What would have happened had we not contacted the customer? What's the root cause of this quality issue? By not addressing these types of questions, data scientists and ML engineers ignore a large fraction of potential beneficiaries of data-informed decisions.

While many of these questions might require vastly different approaches than those referenced above (in particular, for [causal inference](https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/)), **interpretable machine learning** is becoming increasingly important to improve our understanding of what our tried and tested methods actually learn.

Today, there are great resources for interpretable machine learning available online, such as the [Interpretable Machine Learning Book](https://christophm.github.io/interpretable-ml-book/) and various open-source tools ([Shap](https://github.com/slundberg/shap), [Lime](https://github.com/marcotcr/lime), [eli5](https://eli5.readthedocs.io/en/latest/), [interpret](https://github.com/interpretml/interpret), [stratx](https://github.com/parrt/stratx) or scikit-learn's [inspection module](https://scikit-learn.org/stable/inspection.html), just to name a few in the Python universe).

This project builds on top of these methods and implementations to stress test and compare them on real-world data sets. By doing so, I hope to create a resource that can **facilitate our understanding of how common interpretability methods work, where they are helpful and where they might fail**.

# Datasets

## Car Insurance Cold Calls
* Source: [Kaggle](https://www.kaggle.com/kondla/carinsurance)
* Sales calls to insurance customers
* Binary classification: Did the customer end up buying a car insurance or not?
* A few, easy-to-understand features on the customers being called and previous interactions

# Dependencies

You can create a new conda environment containing the required dependencies by running `conda env create -f environment.yml`

# Try it yourself

Before you run the app, make sure that:
1. You have the dataset you would like to use available in `data/raw`
2. You have the `interpret` environment activated

Then, just run `streamlit run src/app.py` from the root of the directory and the app will be served at `localhost:8501`.
