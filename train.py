"""Reproducible evaluation of the included historical valuation dataset."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import GridSearchCV,KFold,train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parent
TARGET="Y house price of unit area"

def load_data(path):
    frame=pd.read_csv(path)
    features=[c for c in frame.columns if c.startswith("X")]
    if len(features)!=6 or TARGET not in frame or len(frame)<20:
        raise ValueError("Expected six X-prefixed features, the documented target and at least 20 rows")
    for column in features+[TARGET]:frame[column]=pd.to_numeric(frame[column],errors="raise")
    if frame[TARGET].isna().any() or not np.isfinite(frame[features+[TARGET]].to_numpy()).all():
        raise ValueError("Features and target must be finite numeric values")
    if "No" in frame and frame["No"].duplicated().any():raise ValueError("Duplicate source record ID")
    return frame,features

def metrics(actual,predicted):
    return {"mae":float(mean_absolute_error(actual,predicted)),
            "rmse":float(np.sqrt(mean_squared_error(actual,predicted))),
            "r2":float(r2_score(actual,predicted))}

def run(data=ROOT/"Real estate.csv",output=ROOT/"reports"):
    frame,features=load_data(data)
    train,test=train_test_split(frame.index,test_size=.3,random_state=50)
    x,y=frame[features],frame[TARGET]
    baseline=DummyRegressor(strategy="median").fit(x.loc[train],y.loc[train])
    search=GridSearchCV(make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),Ridge()),
      {"ridge__alpha":[.1,1,10,100]},scoring="neg_mean_absolute_error",
      cv=KFold(5,shuffle=True,random_state=50),n_jobs=1)
    search.fit(x.loc[train],y.loc[train])
    predictions=search.predict(x.loc[test]);base=baseline.predict(x.loc[test])
    result={"dataset_rows":len(frame),"train_rows":len(train),"holdout_rows":len(test),
      "split":{"method":"random holdout","test_fraction":.3,"random_state":50},
      "selection":{"method":"five-fold training-only CV","metric":"MAE","alpha":search.best_params_["ridge__alpha"],
       "cv_mae":float(-search.best_score_)},"units":"10,000 New Taiwan dollars per ping",
      "holdout":{"median_baseline":metrics(y.loc[test],base),"ridge":metrics(y.loc[test],predictions)},
      "feature_columns":features,"limitations":["Historical Taiwan data, not a current UK or Indian valuation model",
        "Random holdout estimates interpolation performance; no temporal or geographic deployment validation"]}
    output=Path(output);output.mkdir(exist_ok=True,parents=True)
    (output/"metrics.json").write_text(json.dumps(result,indent=2))
    pd.DataFrame({"source_row":test,"actual":y.loc[test],"predicted":predictions,
      "baseline":base,"residual":y.loc[test]-predictions}).to_csv(output/"holdout_predictions.csv",index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,5))
    ax.scatter(y.loc[test],predictions,alpha=.65,color="#087f83",s=25)
    lo=min(y.loc[test].min(),predictions.min());hi=max(y.loc[test].max(),predictions.max())
    ax.plot([lo,hi],[lo,hi],color="#657582",linestyle="--")
    ax.set(xlabel="Observed price per unit area",ylabel="Predicted price per unit area",
      title="Historical valuation holdout\nUnits: 10,000 NTD per ping")
    fig.tight_layout();fig.savefig(output/"holdout.png",dpi=160);plt.close(fig)
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--data",type=Path,default=ROOT/"Real estate.csv")
    p.add_argument("--output",type=Path,default=ROOT/"reports");a=p.parse_args()
    print(json.dumps(run(a.data,a.output),indent=2))
