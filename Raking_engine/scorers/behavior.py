import os
import sys
from datetime import date, datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.constants import (
    BEHAVIOR_RESPONSE_TIME_CEILING_HOURS,
    BEHAVIOR_NOTICE_PERIOD_CEILING_DAYS
)


def get_behavior_score(candidate: dict) -> float:
    """
    Measures how SERIOUS, RELIABLE, and ENGAGED a candidate is in their job search.
    Uses 6 signals from redrob_signals — a root redesign from the original 2-signal version.

    Signals & Weights:
      open_to_work_flag         (0.25) — actively seeking
      interview_completion_rate (0.20) — follow-through / reliability
      github_activity_score     (0.20) — real engineering proof-of-work
      avg_response_time_hours   (0.15) — responsiveness (faster = higher score)
      skill_assessment_scores   (0.10) — platform-verified skill proof
      notice_period_days        (0.10) — availability (shorter = higher score)
    """
    signals = candidate.get('redrob_signals', {})
    if not signals:
        return 0.0

    # 1. Open to work — binary intent signal (weight: 0.25)
    open_to_work = 1.0 if signals.get('open_to_work_flag', False) else 0.0

    # 2. Interview completion rate — direct 0-1 float (weight: 0.20)
    interview_rate = float(signals.get('interview_completion_rate', 0.0))
    interview_rate = max(0.0, min(1.0, interview_rate))

    # 3. GitHub activity score — 0-100 int, -1 means no GitHub linked (weight: 0.20)
    raw_github = signals.get('github_activity_score', -1)
    github_score = 0.0 if raw_github < 0 else min(1.0, float(raw_github) / 100.0)

    # 4. Response time — faster is better; 72h ceiling maps to 0.0 (weight: 0.15)
    raw_response_hours = float(signals.get('avg_response_time_hours', BEHAVIOR_RESPONSE_TIME_CEILING_HOURS))
    response_speed = max(0.0, 1.0 - (raw_response_hours / BEHAVIOR_RESPONSE_TIME_CEILING_HOURS))

    # 5. Skill assessment scores — avg of all platform-verified scores (weight: 0.10)
    assessment_dict = signals.get('skill_assessment_scores', {})
    if assessment_dict and isinstance(assessment_dict, dict):
        scores = [v for v in assessment_dict.values() if isinstance(v, (int, float))]
        skill_assessment_avg = (sum(scores) / len(scores) / 100.0) if scores else 0.0
    else:
        skill_assessment_avg = 0.0

    # 6. Notice period — shorter is better for the hiring team; 90d ceiling maps to 0.0 (weight: 0.10)
    raw_notice = float(signals.get('notice_period_days', BEHAVIOR_NOTICE_PERIOD_CEILING_DAYS))
    notice_score = max(0.0, 1.0 - (raw_notice / BEHAVIOR_NOTICE_PERIOD_CEILING_DAYS))

    behavior_score = (
        open_to_work        * 0.25 +
        interview_rate      * 0.20 +
        github_score        * 0.20 +
        response_speed      * 0.15 +
        skill_assessment_avg * 0.10 +
        notice_score        * 0.10
    )

    return float(max(0.0, min(1.0, behavior_score)))
