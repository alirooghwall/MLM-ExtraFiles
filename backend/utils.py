def build_prompt(user):
    return f"""
    You are a Dari-speaking MLM coach.
    User progress: {user['progress']}
    Checklist: {user['checklist']}
    """