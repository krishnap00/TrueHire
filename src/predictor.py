def predict_job(text):

    text = text.lower()

    # Temporary rule logic (until ML model added)
    fake_keywords = ["urgent", "limited offer", "quick money"]

    fake_score = 0

    for word in fake_keywords:
        if word in text:
            fake_score += 1

    # Convert score to probabilities
    fake_probability = min(fake_score * 0.3, 0.9)
    real_probability = 1 - fake_probability

    if fake_probability > real_probability:
        prediction = "Fake"
    else:
        prediction = "Real"

    return {
        "prediction": prediction,
        "fake_probability": round(fake_probability * 100, 2),
        "real_probability": round(real_probability * 100, 2)
    }