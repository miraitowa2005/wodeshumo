from predict_runtime import RuntimePredictor

try:
    predictor = RuntimePredictor()
    # Test case: 50% SOC, 25C, 10W load
    mins, hist = predictor.predict_remaining_time(0.5, 25.0, 10.0)
    print(f"Test Success: Predicted {mins:.2f} minutes")
except Exception as e:
    print(f"Test Failed: {e}")
