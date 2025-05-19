from typing import Dict

def grade_answer_calibration(model_answer: int, true_passrate: float, eps: int = 10) -> bool:
    """
    Check if the model's answer is within eps% of the true passrate.
    """
    if model_answer < true_passrate - eps or model_answer > true_passrate + eps:
        return False
    return True


def compute_calibration_reward(completion: str, verification_info: Dict) -> int:
    model_response = completion
    true_passrate = verification_info["passrate"]

    # Extract solution.
    if "</think>" in model_response:
        model_solution = model_response.split("</think>")[1]
    else:
        return 0

    # Model must answer with passrate between 0 and 100
    # TODO: edit this if answer formatting instructions change in prompt
    try:
        model_answer = int(model_solution)
        assert 0 <= model_answer <= 100
    except ValueError:
        return 0
    except AssertionError:
        return 0

    if true_passrate is None:
        return 0

    is_correct = grade_answer_calibration(model_answer, true_passrate)
    if is_correct:
        return 1

    return 0
