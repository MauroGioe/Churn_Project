# region Importing libraries
#General use
import pandas  as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sn
#Piplines and data preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
#Correlation and clustering
from dython.nominal import compute_associations
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import fcluster
#Tuning
from sklearn.model_selection import GridSearchCV
#Optimal threshold
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_curve
#Predictions
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score, make_scorer
#Permutaion feature importance
from sklearn.inspection import permutation_importance
from rfpimp import importances
from rfpimp import plot_importances
#PDP
from pdpbox import info_plots
# endregion

# region  Data loading and cleaning
df = pd.read_csv('./DATA/Retention Data/Telco-Customer-Churn.csv',sep=",")
df=df.replace(r'^\s*$', np.nan, regex=True)
#df.isnull().sum()
#Correcting some data-entry errors
#New clients (tenure=0) have missings in total charges, the correct value can be recoverd from the monthly charge
df["TotalCharges"]=df["TotalCharges"].astype(str).astype(float)
df.TotalCharges.fillna(df.MonthlyCharges, inplace=True)
#df[df["tenure"]==0]
df=df.drop("customerID",axis=1)
y=df.Churn
X=df.drop("Churn",axis=1)
#y[y=="Yes"].count()/len(y)
# endregion


# region  Pipelines
categorical_cols = ['gender','SeniorCitizen','Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                      'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', ]

numeric_cols = ["tenure","MonthlyCharges","TotalCharges"]


categorical_pipe = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numeric_pipe = Pipeline([
    ('scaler', StandardScaler())
])


X_preprocessing = ColumnTransformer(
    [('cat', categorical_pipe, categorical_cols),
     ("num", numeric_pipe ,numeric_cols)
    ])


rf = Pipeline([
    ('classifier', RandomForestClassifier(random_state=123,n_estimators=200,class_weight="balanced"))
])
# endregion

# region Data splitting & classification metric
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3,random_state=123)
custom_scorer = make_scorer(fbeta_score, beta=2, pos_label="Yes")
# endregion

# region Data preprocessing
X_train_ori=X_train
X_train=X_preprocessing.fit_transform(X_train)
X_cols=X_preprocessing.transformers_[0][1].named_steps["onehot"].get_feature_names(categorical_cols).tolist()+numeric_cols
X_train=pd.DataFrame(X_train, columns=X_cols)

X_test=X_preprocessing.fit_transform(X_test)
X_cols=X_preprocessing.transformers_[0][1].named_steps["onehot"].get_feature_names(categorical_cols).tolist()+numeric_cols
X_test=pd.DataFrame(X_test, columns=X_cols)
# endregion

# region Correlation & clustering
#The random forest works without problems with correlated features but in order to better asses the features importance
#it is importante to run a groupiwse permutation on the features, thus one has to create clusters of correlated features.
#The Dython library allows to compute association measures for a dataset made up by variables of diffent type.
#Pearson's r for quantitative variables pairs, correlation ratio for quantitative/nominal pairs and Cramers'V for nominals variables as default(or
# Theil's U)
dissimilarity=1-abs(compute_associations(X_train))
diss_condensed =squareform(dissimilarity)
diss_linkage = linkage(diss_condensed, method='complete')

fig, (ax1) = plt.subplots()
ax1.set(xlabel="Features", ylabel="Dissimilarity [0,1]")
plt.axhline(y=0.4, color='r', linestyle='--')
dendro = hierarchy.dendrogram(diss_linkage, labels=list(X_train.columns.values), ax=ax1,leaf_rotation=90)
plt.tight_layout()
#plt.savefig('Clusters.png', bbox_inches='tight')
#pickle.dump(fig,open("Clusters.pickle","wb"))

# endregion

# region Creation clusters
clusters=fcluster(diss_linkage,0.4,criterion='distance')
cluster_output = pd.DataFrame({'Feature':X_train.columns.values.tolist() , 'cluster':clusters})

cluster_feature=cluster_output.groupby('cluster')['Feature'].apply(list)
# endregion


# region Tuning & Fitting
param_grid = {"classifier__max_features": list(range(4, 8)),
              "classifier__max_depth":list([4,12,23,46])}
gs = GridSearchCV(rf, param_grid, cv=7,scoring=custom_scorer)
gs.fit(X_train, y_train)
best_rf=gs.best_estimator_
# endregion

# region Save model
#pickle.dump(gs.best_estimator_,open("best_rf.pickle","wb"))
# endregion

# region Load model
import pickle
best_rf = pickle.load(open("best_rf.pickle","rb"))
# endregion

# region Results

#region optimal threshold
#Defining a function to identify through cross-validation the threshold that maximizes the F2 score
def scoring(y_true, y_proba, random_seed=None,verbose=True):

    def threshold_search(y_true, y_proba):
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba,pos_label="Yes")
        thresholds = np.append(thresholds, 1.001)
        F_2 = 5* recall * precision/(4*precision+recall)
        best_score = np.max(F_2)
        best_th = thresholds[np.argmax(F_2)]
        return best_th

    kf=KFold(n_splits=5,shuffle=True,random_state=random_seed)
    scores = []
    ths = []
    for train_index, test_index in kf.split(y_true, y_true):
        y_prob_train, y_prob_test = y_proba[train_index], y_proba[test_index]
        y_true_train, y_true_test = y_true[train_index], y_true[test_index]
        # determine best threshold on 'train' part
        best_threshold = threshold_search(y_true_train, y_prob_train)
        # use this threshold on 'test' part for score
        opt_predictions = y_prob_test >= best_threshold
        opt_predictions = np.where(opt_predictions == True, "Yes", "No")
        sc = fbeta_score(y_true_test,opt_predictions,beta=2,pos_label="Yes")
        scores.append(sc)
        ths.append(best_threshold)

    best_th = np.mean(ths)
    score = np.mean(scores)

    if verbose: print(f'Best threshold: {np.round(best_th, 4)}, Score: {np.round(score, 5)}')

    return best_th, score

opt_ths=scoring(y_train.to_numpy(),best_rf.predict_proba(X_train)[:,1],random_seed=123)[0]
#Optimal threshold: 0.3791, Train Score: 0.7452
# endregion

# region Predicitons and F2 score

predict_proba=best_rf.named_steps["classifier"].predict_proba(X_test)
opt_predictions = (predict_proba[:,1]>=opt_ths)
opt_predictions=np.where(opt_predictions == True, "Yes", "No")
fbeta_score(y_test,opt_predictions,beta=2,pos_label="Yes")
#F2 score= 0.744
conf_mat = confusion_matrix(y_test, opt_predictions)
#print(conf_mat)

# endregion

# endregion

# region Charts
# region Plot conf matrix
fig, ax= plt.subplots()
sn.heatmap(conf_mat,annot=True, fmt="d")
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['Client', 'Ex-client']); ax.yaxis.set_ticklabels(['Client', 'Ex-client']);

#plt.savefig('Conf_matrix.png', bbox_inches='tight')
#pickle.dump(fig,open("Conf_matrix.pickle","wb"))

# endregion

# region Churn related metrics
#Computing quantities for the lift/gain charts

df_pred_proba=pd.DataFrame({"Predicted_churn_probability": predict_proba[:,1]})
df_pred_proba=df_pred_proba.set_index(y_test.index)
deciles=pd.cut(df_pred_proba.Predicted_churn_probability, np.percentile(df_pred_proba.Predicted_churn_probability,list(range(0,101,10))),include_lowest=True)
deciles=deciles.rename("Interval")
deciles=pd.concat([deciles,y_test],axis=1)
deciles_count=deciles[deciles["Churn"]=="Yes"].groupby("Interval").count().values[:,0]
deciles_count=deciles_count[::-1]
cumulative_churn_rate=np.cumsum(deciles_count)/np.sum(deciles_count)*100
cumulative_sum_clients=np.cumsum(deciles.groupby("Interval").count())/len(deciles)*100
cumulative_sum_clients=cumulative_sum_clients.rename(columns={"Churn":"Percentage of clients"})
lift_score=cumulative_churn_rate/cumulative_sum_clients.values.T
true_churn_rate=deciles[deciles["Churn"]=="Yes"].groupby("Interval").count().values[:,0]/deciles.groupby("Interval").size()
average_churn_rate=np.round(y[y=="Yes"].count()/len(y),2)
# endregion

# region Lift chart
plt.plot(list(range(1,11)),lift_score.T)
fig, ax = plt.subplots()
sn.regplot(list(range(1,11)),lift_score,lowess=True)
ax.set_xticks(np.arange(1,11))
x_labs=list(["10%","20%","30%","40%","50%","60%","70%","80%","90%","100%"])
ax.set_xticklabels(x_labs)
ax.plot([0.8,10.2 ], [1, 1],color="black")
plt.xlim(0.8,10.2)
ax.set(xlabel="Predicted probabilities deciles",ylabel="Lift")
plt.text(2,1.05,'Random selection')


#plt.savefig('Lift_chart.png', bbox_inches='tight')
#pickle.dump(fig,open("Lift_chart.pickle","wb"))



#sujects who really dropped in the deciles of the predicted probabilities
#Estimated probability seem accurate only for the tenth decile
#Otherwise it seems to overestimate the acutal churn rate
fig, ax = plt.subplots()
plt.bar(list(range(1,11)),true_churn_rate)
ax.set_xticks(np.arange(1,11))
x_labs=list(["10%","20%","30%","40%","50%","60%","70%","80%","90%","100%"])
ax.set_xticklabels(x_labs)
ax.plot([0.6,10.4 ], [average_churn_rate, average_churn_rate],color="black")
plt.xlim(0.5,10.5)
plt.text(1,average_churn_rate+0.005,'Average churn rate')
plt.axvline(x=5,color="orange")
#Groups over the probability threshold
plt.text(5.015,0.5,'Prediction threshold')
ax.set(xlabel="Predicted probabilities deciles",ylabel="True churn rate")
fig.tight_layout()

#plt.savefig('Lift_chart2.png', bbox_inches='tight')
#pickle.dump(fig,open("Lift_chart2.pickle","wb"))
# endregion

# region Cumulative gain
fig, ax = plt.subplots()
sn.regplot(cumulative_sum_clients,cumulative_churn_rate,lowess=True,label="Lift curve")
plt.xlim(9,102)
plt.ylim(9,102)
ax.set(xlabel="Clients (%)",ylabel="Former clients at the end of the month (%)")
ax.set_xticks(np.arange(10,101,10))
ax.set_yticks(np.arange(10,101,10))
x_labs=list(["10%","20%","30%","40%","50%","60%","70%","80%","90%","100%"])
ax.set_xticklabels(x_labs)
ax.set_yticklabels(x_labs)
random_sel = [10 * i + 10 for i in range(0,10)]
ax.plot(cumulative_sum_clients,random_sel,label="Baseline")
plt.legend(loc="upper left")

#plt.savefig('Cumulative_gain.png', bbox_inches='tight')
#pickle.dump(fig,open("Cumulative_gain.pickle","wb"))

# endregion


# region Group permutaion
#To get an output which is easier to read (importance not splitted across correlated features) one should use either conditional permutation importance
#(not available in python) or resort to use the library rfpimp to compute group permutations.

np.random.seed(123)
group_imp=importances(best_rf.named_steps["classifier"],X_test,y_test,features=list(cluster_feature),metric=custom_scorer)
fig, ax = plt.subplots()
ax.set(xlabel="Drop in $F_2$ score when the variable is perturbed")
plot_importances(group_imp,ax=ax)
plt.xticks(np.arange(min(group_imp.values), max(group_imp.values)+0.03, 0.01))
fig.set_size_inches(10,10)
ax.set_xlim([0, 0.10])
fig.tight_layout()




#plt.savefig('Feature_importance_group.png', bbox_inches='tight')
#pickle.dump(fig,open("Feature_importance_group.pickle","wb"))
# endregion


# region Predictions boxplots
#Variables with high importance in the predicton:
# Internet service/Monthly charges
#Lenght of contract
#Tenure
#Online security
#Tech support

#defining a small function to unscale the x axis
def unscaling(x_scaled,X_col_unsc):
    return ( np.round((x_scaled* X_col_unsc.std(axis=0)) + X_col_unsc.mean(axis=0),1).astype(int) )

# region Tenure

#Actual plot
fig, axes, summary_df =info_plots.actual_plot(model=best_rf.named_steps["classifier"], X=X_train,
                             feature="tenure",feature_name="Tenure",predict_kwds={},num_grid_points=11)

axes["bar_ax"].set_xticklabels(pd.qcut(X_train_ori.tenure,10,precision=3).values.categories.values)



#plt.savefig('Predi_boxplot_tenure.png', bbox_inches='tight')
#pickle.dump(fig,open("Predi_boxplot_tenure.pickle","wb"))

# endregion

# region Internet


fig, axes, summary_df =info_plots.actual_plot(model=best_rf.named_steps["classifier"], X=X_train,
                             feature=["InternetService_DSL","InternetService_Fiber optic","InternetService_No"],feature_name="Internet service",predict_kwds={},
                                              num_grid_points=11 )
axes["bar_ax"].set_xticklabels(["DSL","Optical fiber","No internet"])

#plt.savefig('Predi_boxplot_internet.png', bbox_inches='tight')
#pickle.dump(fig,open("Predi_boxplot_internet.pickle","wb"))

# endregion

# region Montlhy charges

fig, axes, summary_df =info_plots.actual_plot(model=best_rf.named_steps["classifier"], X=X_train,
                             feature="MonthlyCharges",feature_name="MonthlyCharges",predict_kwds={},num_grid_points=11)

axes["bar_ax"].set_xticklabels(pd.qcut(X_train_ori.MonthlyCharges,10,precision=3).values.categories.values)

#Compute mean and SD to evaluate the price of the optical fiber
#X_train_ori.MonthlyCharges.mean()
#X_train_ori.MonthlyCharges.std()
#Fiber is costly/not good enough for that price
#X_train_ori.MonthlyCharges[X_train_ori.InternetService=="Fiber optic"].mean()
#X_train_ori.MonthlyCharges[X_train_ori.InternetService=="Fiber optic"].std()
#plt.savefig('Predi_boxplot_montlhy_charges.png', bbox_inches='tight')
#pickle.dump(fig,open("Predi_boxplot_montlhy_charges.pickle","wb"))
# endregion


# region Contract
fig, axes, summary_df =info_plots.actual_plot(model=best_rf.named_steps["classifier"], X=X_train,
                             feature=["Contract_Month-to-month","Contract_One year","Contract_Two year"],feature_name="Contract",predict_kwds={})
axes["bar_ax"].set_xticklabels(["Month-to-month","One year","Two year"])
#plt.savefig('Predi_boxplot_contract.png', bbox_inches='tight')
#pickle.dump(fig,open("Predi_boxplot_contract.pickle","wb"))
# endregion

# region Techsupport

fig, axes, summary_df  = info_plots.actual_plot(model=best_rf.named_steps["classifier"], X=X_train,
                             feature=["TechSupport_No","TechSupport_Yes"],feature_name="Tech support",predict_kwds={})

axes["bar_ax"].set_xticklabels(["No","Yes"])

#pickle.dump(fig,open("Predi_boxplot_techsupport.pickle","wb"))
# endregion

# region Online security

fig, axes, summary_df  = info_plots.actual_plot(model=best_rf.named_steps["classifier"], X=X_train,
                             feature=["OnlineSecurity_No","OnlineSecurity_Yes"],feature_name="Online security",predict_kwds={})

axes["bar_ax"].set_xticklabels(["No","Yes"])

#pickle.dump(fig,open("Predi_boxplot_Online_security.pickle","wb"))
# endregion

# endregion

# endregion