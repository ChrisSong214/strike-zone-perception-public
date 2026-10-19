# Exploratory diagnostics only; not an end-to-end prospective validation.
# Reads existing CSVs and SQLite in read-only mode; writes only its results JSON.
# Use a known-working NumPy/Pandas runtime; no project model inference is performed.
import pathlib,sqlite3,json
import numpy as np
import pandas as pd
r=pathlib.Path(__file__).resolve().parents[2]
frames=[]
for p in sorted((r/'data/leaderboards').glob('*_dva_leaderboard.csv')):
 d=pd.read_csv(p);frames.append(d)
rates=pd.concat(frames,ignore_index=True).drop_duplicates(['batter_id','season'])
print('RATINGS',rates.groupby('season').size().to_dict())
sub=rates[rates.n_pitches>=750].copy()
print('COMPONENT_CORRELATION',sub[['tendency_dva','actual_dva']].corr().iloc[0,1])
def fit_predict(train,test,cols,target):
 a=np.column_stack([np.ones(len(train)),train[cols].to_numpy(float)])
 b=np.linalg.lstsq(a,train[target].to_numpy(float),rcond=None)[0]
 return np.column_stack([np.ones(len(test)),test[cols].to_numpy(float)])@b
# Can elementary discipline rates reproduce DVA in a held-out season?
train=sub[sub.season<=2024];test=sub[sub.season==2025]
yhat=fit_predict(train,test,['chase_rate','zone_swing_rate'],'swing_rating');y=test.swing_rating.to_numpy()
print('HELDOUT_2025_DVA_FROM_CHASE_ZONE_R2',1-((y-yhat)**2).sum()/((y-y.mean())**2).sum(),'N',len(y))
c=sqlite3.connect('file:'+str(r/'data/mlb.db')+'?mode=ro',uri=True)
out=pd.read_sql_query('''SELECT batter AS batter_id,game_year AS season,COUNT(*) AS pa,
SUM(woba_value)*1.0/SUM(woba_denom) AS woba
FROM statcast WHERE events IS NOT NULL AND events != '' GROUP BY batter,game_year''',c);c.close()
d=sub.merge(out,on=['batter_id','season']);d=d[d.pa>=197].copy();d['log_n_pitches']=np.log(d.n_pitches)
future=d[['batter_id','season','woba']].copy();future['season']=future.season-1;future=future.rename(columns={'woba':'future_woba'})
pairs=d.merge(future,on=['batter_id','season']);pairs['target_year']=pairs.season+1
base=['woba','chase_rate'];strong=base+['zone_swing_rate','log_n_pitches'];results=[]
# 2023, 2024, 2025 held out at the aggregate forecasting stage only.
for yr in [2023,2024,2025]:
 tr=pairs[pairs.target_year<yr];te=pairs[pairs.target_year==yr].copy()
 for name,cols in [('base',base),('base_dva',base+['swing_rating']),('strong',strong),('strong_dva',strong+['swing_rating'])]:te[name]=fit_predict(tr,te,cols,'future_woba')
 results.append(te);print('FOLD',yr,'TRAIN',len(tr),'TEST',len(te))
pred=pd.concat(results,ignore_index=True);y=pred.future_woba.to_numpy();summary={}
for name in ['base','base_dva','strong','strong_dva']:
 err=(y-pred[name].to_numpy())**2;summary[name]={'r2':float(1-err.sum()/((y-y.mean())**2).sum()),'rmse':float(np.sqrt(err.mean()))}
print('TEMPORAL_AGGREGATE_FORECAST',json.dumps(summary))
folds=[]
for yr,g in pred.groupby('target_year'):
 vals={k:float(np.sqrt(np.mean((g.future_woba-g[k])**2))) for k in summary};folds.append({'year':int(yr),'n':len(g),'rmse':vals})
print('FOLD_RMSE',json.dumps(folds))
# Cluster by hitter when uncertainty is summarized across repeated seasons; fitted forecasts fixed.
for a,b in [('base','base_dva'),('strong','strong_dva')]:
 dif=(pred.future_woba-pred[a])**2-(pred.future_woba-pred[b])**2
 by=pd.DataFrame({'batter':pred.batter_id,'diff':dif}).groupby('batter')['diff'].agg(['sum','count'])
 rng=np.random.default_rng(20261018);samples=[]
 for _ in range(2000):
  idx=rng.integers(0,len(by),len(by));samples.append(by['sum'].to_numpy()[idx].sum()/by['count'].to_numpy()[idx].sum())
 print('MSE_REDUCTION',a,b,float(dif.mean()),'CLUSTER_CI95',np.quantile(samples,[.025,.975]).tolist())
(r/'docs/presentation/research_idea_pilot_results.json').write_text(json.dumps({'n':len(pred),'component_correlation':float(sub[['tendency_dva','actual_dva']].corr().iloc[0,1]),'dva_reconstruction_r2':float(1-((yhat-test.swing_rating.to_numpy())**2).sum()/((test.swing_rating-test.swing_rating.mean())**2).sum()),'forecasts':summary,'folds':folds},indent=2))
