class HardFilter:
    def __init__(self, behavior_df):
        self.behavior_df = behavior_df

    def is_rejected(self, candidate: dict) -> bool:
        cid = candidate.get('candidate_id')
        if not cid or cid not in self.behavior_df.index:
            return True # Reject unknown
            
        row = self.behavior_df.loc[cid]
        
        label = row.get('honey_pot_labels', 'trusted')
        if label in ('suspicious', 'high_risk'):
            return True
            
        hp_score = row.get('honeypot_score', 0)
        if isinstance(hp_score, (int, float)) and hp_score >= 3:
            return True
            
        date_anomaly = row.get('date_anomaly', False)
        is_anomaly = False
        
        if isinstance(date_anomaly, bool):
            is_anomaly = date_anomaly
        elif isinstance(date_anomaly, str):
            is_anomaly = date_anomaly.lower() in ('true', '1', 'yes')
        elif isinstance(date_anomaly, (int, float)):
            is_anomaly = date_anomaly == 1
            
        if is_anomaly:
            return True
            
        return False
