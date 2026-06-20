import os
import sys
from datetime import date, datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.constants import TRUST_ENDORSEMENTS_MAX, TRUST_RECENCY_WINDOW_DAYS


def get_trust_score(candidate: dict) -> float:
    """
    Measures how LEGITIMATE, CREDIBLE, and TRUSTWORTHY a profile is.
    Uses 6 signals from redrob_signals — a root redesign from the original 2-signal version.

    Signals & Weights:
      profile_completeness_score (0.25) — how filled-in the profile is
      verified_email + phone     (0.25) — identity verification (both = 1.0, one = 0.5, none = 0.0)
      recruiter_response_rate    (0.20) — professionalism with recruiters
      last_active_date           (0.15) — recency (active today = 1.0, 1yr+ ago = 0.0)
      endorsements_received      (0.10) — peer-validated credibility
      linkedin_connected         (0.05) — professional identity linkage
    """
    signals = candidate.get('redrob_signals', {})
    if not signals:
        return 0.0

    # 1. Profile completeness — direct 0-100 value (weight: 0.25)
    completeness = float(signals.get('profile_completeness_score', 0)) / 100.0
    completeness = max(0.0, min(1.0, completeness))

    # 2. Identity verification — email + phone each contribute 0.5 (weight: 0.25)
    email_verified = 1.0 if signals.get('verified_email', False) else 0.0
    phone_verified = 1.0 if signals.get('verified_phone', False) else 0.0
    identity_verified = (email_verified + phone_verified) / 2.0

    # 3. Recruiter response rate — direct 0-1 float (weight: 0.20)
    response_rate = float(signals.get('recruiter_response_rate', 0.0))
    response_rate = max(0.0, min(1.0, response_rate))

    # 4. Last active date recency — linear decay over TRUST_RECENCY_WINDOW_DAYS (weight: 0.15)
    recency = 0.0
    last_active_raw = signals.get('last_active_date', None)
    if last_active_raw:
        try:
            last_active = datetime.strptime(str(last_active_raw), '%Y-%m-%d').date()
            days_inactive = (date.today() - last_active).days
            recency = max(0.0, 1.0 - (days_inactive / TRUST_RECENCY_WINDOW_DAYS))
        except (ValueError, TypeError):
            recency = 0.0

    # 5. Endorsements received — capped at TRUST_ENDORSEMENTS_MAX (weight: 0.10)
    endorsements = int(signals.get('endorsements_received', 0))
    endorsement_score = min(1.0, endorsements / TRUST_ENDORSEMENTS_MAX)

    # 6. LinkedIn connected — binary professional identity proof (weight: 0.05)
    linkedin_score = 1.0 if signals.get('linkedin_connected', False) else 0.0

    trust_score = (
        completeness       * 0.25 +
        identity_verified  * 0.25 +
        response_rate      * 0.20 +
        recency            * 0.15 +
        endorsement_score  * 0.10 +
        linkedin_score     * 0.05
    )

    return float(max(0.0, min(1.0, trust_score)))
