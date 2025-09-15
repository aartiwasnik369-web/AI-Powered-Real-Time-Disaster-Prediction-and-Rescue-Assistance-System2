# backend/decision_engine.py
def make_decision(flood_result: dict, detection_result: dict) -> dict:
    """
    Combines flood risk prediction + object detection to form a disaster decision.
    """
    risk_level = flood_result.get("risk_level", "unknown")
    confidence = flood_result.get("confidence", 0.0)

    detected_objects = detection_result.get("objects", [])
    human_count = sum(1 for obj in detected_objects if obj["label"] == "person")
    animal_count = sum(1 for obj in detected_objects if obj["label"] in ["dog", "cat", "cow"])

    message = "Safe"
    if risk_level == "high" and (human_count > 0 or animal_count > 0):
        message = f"🚨 HIGH RISK! {human_count} humans & {animal_count} animals detected. Immediate evacuation required!"
    elif risk_level == "medium":
        message = f"⚠️ Medium risk. Stay alert. Detected: {human_count} humans, {animal_count} animals."
    elif risk_level == "low":
        message = "✅ Low risk. Situation normal."

    return {
        "risk_level": risk_level,
        "confidence": confidence,
        "human_count": human_count,
        "animal_count": animal_count,
        "message": message
    }
