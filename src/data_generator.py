import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_data(num_records=100):
    """
    Generates synthetic temporal data simulating a user's sharing history.
    The data includes:
- Timestamps that become more frequent towards the end (simulating increased velocity).
- Text that starts neutral and becomes more polarized/negative towards the end (simulating NLP stance shifts).  
    """
    np.random.seed(42)
    start_time = datetime.now() - timedelta(days=7)
    
    data = []
    for i in range(num_records):
        # Simulate time passing (increasing velocity towards the end)
        time_increment = timedelta(hours=np.random.randint(1, 12) if i < 80 else np.random.randint(1, 3))
        start_time += time_increment
        
        # Simulate text (normal at first, highly polarized/negative later)
        if i < 80:
            text = "Just a normal day, sharing some standard news."
        else:
            text = "OUTRAGEOUS! You won't believe what they are hiding from us! Fake lies everywhere!"
            
        data.append({
            "timestamp": start_time,
            "user_id": "user_101",
            "tweet_text": text
        })
        
    df = pd.DataFrame(data)
    return df