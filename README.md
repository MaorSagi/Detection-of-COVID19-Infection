# Detection of COVID 19 Infection
The goal of this project is to develop a classifier for predict if a patient infected by Covid 19 using regular blood tests data.

Based on the paper of Alves et al., Explaining Machine Learning based Diagnosis of COVID-19 from Routine Blood Tests with Decision Trees and Criteria Graphs. I wrote code that reproduced the experiments in the paper, and then suggested new features and evaluate additional classifiers: CatBoost and LightGBM. The results explained using SHAP.
You can find the paper [here](https://www.sciencedirect.com/science/article/pii/S0010482521001293?via%3Dihub)

<p>Running instructions:</p>
<ol>
<li>Notice your working directory is Snapshot-Ensamble-Network folder. otherwise run the command:</li>
<code>cd Snapshot-Ensamble-Network</code>
<li>Install all the libraries required by running the following command:</li>
<code>pip install -r requirements.txt</code>
<li>In the consts file determine the parameters regarding to your needs.</li>
<li>Run the project by running the command:</li>
<code>python main.py</code>
</ol>
