FROM tensorflow/tensorflow:latest-py3

RUN pip install --upgrade pip && pip install pandas sklearn tqdm matplotlib ipywidgets scikit-learn scikit-image seaborn

RUN groupadd -g 1000 user
RUN useradd -u 1000 -g 1000 -m user && yes password | passwd user

USER user