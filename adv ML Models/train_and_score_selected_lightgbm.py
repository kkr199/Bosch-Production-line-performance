"""Train the selected benchmark configuration and score the full test feature set."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from selection_utils import profile_training_fold, selected_features, validation_threshold, select_categories_on_train  # noqa
from src.data.phase6_predictive_failure_modeling import build_model_dataset, find_dataset_path  # noqa

RATE=.0025; SEED=42; OUT=ROOT/'data'/'processed'; REPORTS=ROOT/'reports'; MODELS=ROOT/'models'
def main():
    target=pd.read_csv(find_dataset_path('train_numeric.csv'),usecols=['Id','Response']); target.Id=target.Id.astype('int64'); target.Response=target.Response.astype('int8')
    selection_train,_=train_test_split(target,test_size=.20,stratify=target.Response,random_state=SEED)
    profile=profile_training_fold(set(selection_train.Id),len(selection_train)); profile['missing_rate']=1-profile.present_rate
    numeric,candidates=selected_features(profile,RATE); cats,levels=select_categories_on_train(selection_train)
    train=build_model_dataset(target,numeric,cats,levels,profile,'train').set_index('Id'); features=[c for c in train if c!='Response']
    y=target.Response.to_numpy(); x=train.reindex(target.Id)[features]
    model=LGBMClassifier(n_estimators=250,learning_rate=.04,num_leaves=48,subsample=.85,colsample_bytree=.85,class_weight='balanced',random_state=SEED,n_jobs=4,verbose=-1)
    model.fit(x,y); cutoff=validation_threshold(model.predict_proba(x)[:,1],y)
    test_ids=pd.read_csv(find_dataset_path('test_numeric.csv'),usecols=['Id']); test_ids.Id=test_ids.Id.astype('int64')
    test=build_model_dataset(test_ids,numeric,cats,levels,profile,'test').set_index('Id')
    for c in features:
        if c not in test: test[c]=0
    scores=model.predict_proba(test.reindex(test_ids.Id)[features])[:,1]
    predictions=pd.DataFrame({'Id':test_ids.Id,'failure_probability':scores,'predicted_failure_alert':(scores>=cutoff).astype('uint8')})
    OUT.mkdir(parents=True,exist_ok=True); MODELS.mkdir(parents=True,exist_ok=True)
    predictions.to_parquet(OUT/'advanced_ml_test_predictions.parquet',index=False,compression='zstd')
    predictions[['Id', 'predicted_failure_alert']].rename(
        columns={'predicted_failure_alert': 'Response'}
    ).to_csv(OUT/'advanced_ml_sample_submission.csv', index=False)
    summary=pd.DataFrame([{'model':'LightGBM','validation_split':'80/20','numeric_present_rate':RATE,'numeric_candidates':candidates,'numeric_features':len(numeric),'categorical_columns':len(cats),'one_hot_levels':sum(map(len,levels.values())),'final_feature_count':len(features),'test_products_scored':len(predictions),'test_alerts':int(predictions.predicted_failure_alert.sum()),'test_alert_rate_pct':float(predictions.predicted_failure_alert.mean()*100),'decision_threshold':float(cutoff)}])
    summary.to_csv(REPORTS/'advanced_ml_test_prediction_summary.csv',index=False)
    predictions.nlargest(25,'failure_probability').to_csv(REPORTS/'advanced_ml_top_test_risk_preview.csv',index=False)
    joblib.dump({'model':model,'feature_cols':features,'threshold':cutoff,'selection_train_ids':selection_train.Id.to_numpy(),'metadata':summary.iloc[0].to_dict()},MODELS/'advanced_ml_lightgbm_80_20_0_0025.joblib')
    (REPORTS/'advanced_ml_model_card.json').write_text(json.dumps(summary.iloc[0].to_dict(),indent=2))
if __name__=='__main__': main()
