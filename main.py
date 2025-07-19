import pykeen
from pykeen.models import TransE, RotatE, TuckER
from pykeen.evaluation import RankBasedEvaluator
from pykeen.triples import TriplesFactory
from pykeen.predict import predict_target
import torch
import optuna
import json
from collections import Counter
import numpy as np

RELATIONS=[
    'click_about',
    'click_courseware',
    'click_forum',
    'click_info',
    'click_progress',
    'close_courseware',
    'create_comment',
    'create_thread',
    'load_video',
    'pause_video',
    'play_video',
    'problem_check',
    'problem_check_correct',
    'problem_check_incorrect',
    'problem_get',
    'problem_save',
    'reset_problem',
    'seek_video',
    'stop_video'
  ]

def load_GE_model(study,train):
    Tucker_best_params = study.best_params
    model = TuckER(
        triples_factory=train,
        embedding_dim=Tucker_best_params['model.embedding_dim'],
        relation_dim=Tucker_best_params['model.relation_dim'],
        dropout_0=Tucker_best_params['model.dropout_0'],
        dropout_1=Tucker_best_params['model.dropout_1'],
        dropout_2=Tucker_best_params['model.dropout_2'],
        entity_initializer=Tucker_best_params['model.entity_initializer'],
        random_seed=42
    )
    states = torch.load("trained_models/TuckER.pt", map_location=torch.device('cpu'))
    model.load_state_dict(states['state_dict'])
    model.eval()
    return model

def load_XGB(study):
  model=XGBClassifier(
      **study.best_params
  )
  model=joblib.load("xgb_model.pkl")
  return model

def closest_tails(model, user, relations):
  inferences=[]
  for r in relations:
    pred=predict_target(model, head=user, relation=r, triples_factory=test)
    inferences.append(pred.df[['tail_id','score']][1:5])
  closest_tails=[]
  for j in range(4):
    scores=[]
    for i in inferences:
      scores.append(i['tail_id'].iloc[j])
    closest_tails.append(Counter(scores).most_common(1)[0][0])
  closest_tails=[int(n) for n in closest_tails]
  return list(set(closest_tails))


def generate_sample(user, relations, course, model):
  ranking = predict_target(model, head=user, tail=course, triples_factory=test)
  ranking_dataframe = pd.DataFrame(ranking.df[['relation_label', 'score']][:8])

  # Create the dict, 1 if the action is in the best ranked actions, 0 otherwise
  output_dict = {key: (1 if key in list(ranking_dataframe['relation_label']) else 0) for key in relations}
  sample_action_values = [output_dict[key] for key in columns]
  sample = [user, course] + sample_action_values
  return(sample)

if __name__ == '__main__':
    data = TriplesFactory.from_path('datasets/triples.tsv')
    complete_train, test = data.split([0.8, 0.2], random_state=42)
    train, val = complete_train.split([0.8, 0.2], random_state=42)
    study_TuckER=optuna.load_study(study_name='TuckER_hpo_study', storage='sqlite:///optuna_studies/optuna_study_TuckER.db')
    study_XGB=optuna.load_study(study_name='XGBoost_Optimization',storage='sqlite:///optuna_studies/XGB__optuna_study.db')
    model_tuckER=load_GE_model(study_TuckER,train)
    model_XGB=load_XGB(study_XGB)
    print('***MODELS LOADED SUCCESSFULLY***')
    user=input('CHOOSE A USER ID(advice: control entities_to_id.json to check existing ids): ')
    closest=closest_tails(model_tuckER,user,RELATIONS)
    print(f'CLOSEST COURSES ID: {closest}')
    sample=generate_sample(user,RELATIONS,closest[0], model_tuckER)
    print(f'GENERATED SAMPLE:{sample}')
    pred=model_XGB.predict(np.array([sample]))
    print(pred) #1 if dropout, 0 otherwise


