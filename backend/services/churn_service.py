def predict_churn(user_data:dict):
    score = 0.2+0.01*user_data['num_pauses']+0.005*user_data['day_active']
    return round(min(score,1.0),3)
#  """dummy logic (replace later with ml model)"""
