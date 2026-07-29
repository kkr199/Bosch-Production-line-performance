"""Complete failed Phase 6 Logistic Regression and Random Forest benchmark rows."""
from __future__ import annotations
import importlib.util
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, matthews_corrcoef, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

OUT=Path(__file__).resolve().parent; ROOT=OUT.parent
sys.path.insert(0,str(ROOT))
from selection_utils import profile_training_fold, selected_features, validation_threshold, select_categories_on_train  # noqa
from src.data.phase6_predictive_failure_modeling import build_model_dataset, find_dataset_path  # noqa

SPLITS={'75/25':.25,'80/20':.20,'90/10':.10}; RATES=(.001,.0025,.005,.01,.02); SEED=42
SOURCE=OUT/'phase6_full_clean_split_rate_metrics.csv'; TARGET=OUT/'phase6_full_clean_split_rate_metrics_completed.csv'

def arrays(train_frame, valid_frame):
    x=np.asarray(train_frame,dtype=np.float32); v=np.asarray(valid_frame,dtype=np.float32)
    med=np.nanmedian(x,axis=0).astype(np.float32); med[np.isnan(med)]=0
    for a in (x,v):
        mask=np.isnan(a); a[mask]=med[np.where(mask)[1]]
    return x,v

def run():
    old=pd.read_csv(SOURCE); completed=old[old.status=='completed'].to_dict('records'); rows=completed[:]
    target=pd.read_csv(find_dataset_path('train_numeric.csv'),usecols=['Id','Response']); target.Id=target.Id.astype('int64'); target.Response=target.Response.astype('int8')
    for label,fraction in SPLITS.items():
        train,valid=train_test_split(target,test_size=fraction,stratify=target.Response,random_state=SEED)
        print(label,flush=True); profile=profile_training_fold(set(train.Id),len(train)); profile['missing_rate']=1-profile.present_rate
        cats,levels=select_categories_on_train(train); y,yv=train.Response.to_numpy(),valid.Response.to_numpy()
        for rate in RATES:
            numeric,candidates=selected_features(profile,rate); frame=build_model_dataset(target,numeric,cats,levels,profile,'train').set_index('Id'); features=[c for c in frame if c!='Response']
            for name in ('Logistic Regression','Random Forest'):
                started=time.perf_counter(); status='completed'; error=''
                try:
                    x,v=arrays(frame.reindex(train.Id)[features],frame.reindex(valid.Id)[features])
                    if name=='Logistic Regression':
                        scaler=StandardScaler(copy=False); x=scaler.fit_transform(x); v=scaler.transform(v); model=LogisticRegression(max_iter=200,class_weight='balanced',solver='lbfgs',random_state=SEED)
                    else: model=RandomForestClassifier(n_estimators=60,max_depth=14,min_samples_leaf=25,class_weight='balanced_subsample',n_jobs=4,random_state=SEED)
                    model.fit(x,y); tr=model.predict_proba(x)[:,1]; scores=model.predict_proba(v)[:,1]; cut=validation_threshold(tr,y); pred=(scores>=cut).astype(np.uint8)
                    mcc=float(matthews_corrcoef(yv,pred)); pr=float(average_precision_score(yv,scores)); precision=float(precision_score(yv,pred,zero_division=0)); recall=float(recall_score(yv,pred,zero_division=0))
                except Exception as exc: status='failed'; error=f'{type(exc).__name__}: {exc}'; mcc=pr=precision=recall=cut=np.nan
                rows.append({'split_ratio':label,'rows':len(target),'train_rows':len(train),'validation_rows':len(valid),'numeric_present_rate':rate,'numeric_candidates':candidates,'numeric_features':len(numeric),'categorical_columns':len(cats),'one_hot_levels':sum(map(len,levels.values())),'final_feature_count':len(features),'model':name,'mcc':mcc,'pr_auc':pr,'precision':precision,'recall':recall,'runtime_seconds':time.perf_counter()-started,'status':status,'error':error})
                pd.DataFrame(rows).to_csv(TARGET,index=False); print(label,rate,name,status,flush=True)
                del x,v
            del frame
    return pd.DataFrame(rows)
if __name__=='__main__': print(run().to_string(index=False))
