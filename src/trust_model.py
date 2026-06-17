import pandas as pd
from textblob import TextBlob

class TrustDecayModel:
    def __init__(self, lambda_factor=0.9, gamma=0.1, w1=0.6, w2=0.4):
        self.lambda_factor = lambda_factor # Memory of past trust
        self.gamma = gamma                 # Overall decay penalty multiplier
        self.w1 = w1                       # Weight for velocity penalty
        self.w2 = w2                       # Weight for NLP stance penalty
        self.base_trust = 100.0            # Starting trust score

    def analyze_stance(self, text):
        """Simple NLP stance detection using TextBlob polarity."""
        analysis = TextBlob(text)
        # Polarity is -1 (negative) to 1 (positive). We penalize extreme negativity/polarization.
        polarity = analysis.sentiment.polarity
        return abs(polarity) # High absolute value = high polarization

    def process_pipeline(self, df):
        """Calculates velocity, NLP stance, and temporal trust decay."""
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        trust_scores = []
        velocities = []
        stances = []
        
        current_trust = self.base_trust
        
        for i in range(len(df)):
            # 1. NLP Stance Shift
            stance = self.analyze_stance(df.iloc[i]['tweet_text'])
            stances.append(stance)
            
            # 2. Temporal Velocity (time difference between last tweet)
            if i == 0:
                velocity = 0
            else:
                time_diff = (df.iloc[i]['timestamp'] - df.iloc[i-1]['timestamp']).total_seconds() / 3600
                # Velocity is higher if time_diff is smaller (spamming)
                velocity = 1.0 / (time_diff + 0.1) 
            velocities.append(velocity)
            
            # 3. Apply Trust Decay Formula
            penalty = self.gamma * ((self.w1 * velocity) + (self.w2 * stance))
            current_trust = (self.lambda_factor * current_trust) - penalty
            
            # Floor the trust score at 0
            current_trust = max(0, current_trust)
            trust_scores.append(current_trust)
            
        df['velocity'] = velocities
        df['stance_penalty'] = stances
        df['trust_score'] = trust_scores
        
        return df