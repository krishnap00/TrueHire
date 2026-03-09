def predict_job(text):
    """
    Temporary prediction logic.
    This will later be replaced with the ML model.
    """

    text = text.lower()

    if "urgent" in text or "limited offer" in text or "quick money" in text:
        return "Fake Job Posting"
    else:
        return "Genuine Job Posting"