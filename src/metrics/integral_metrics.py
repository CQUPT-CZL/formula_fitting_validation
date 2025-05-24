from typing import List, Dict, Any
import inspect
import numpy as np
import random
import logging

class IAE():
    def compute(
        self,
        code: str,
        gt_code: str,
    ) -> float | None:
        try:

            func_pred = get_func_from_code(code)
            func_gt = get_func_from_code(gt_code)

            if func_pred is None or func_gt is None:
                return None

            inputs = generate_inputs(func_gt, n_samples = 1000)
            diffs = []
            for args in inputs:
                try:
                    y_pred = func_pred(*args)
                    y_true = func_gt(*args)
                    diffs.append(abs(y_pred - y_true))
                except Exception as e:
                    # print(f"Error during function execution: {e}")
                    continue

            diffs = [d for d in diffs if np.isfinite(d)]
            iae = sum(diffs) / len(diffs) if diffs else None
            return iae
        except Exception as e:
            logging.error(f"Error computing ICE: {e}")
            raise


def get_func_from_code(code_str):
    safe_globals = {
        "__builtins__": __builtins__,
        "math": __import__("math"),
        "np": np,
    }
    local_vars = {}
    try:
        exec(code_str, safe_globals, local_vars)
        return local_vars.get("calculate_value")
    except Exception:
        return None


def generate_inputs(func, n_samples=1000, input_range=(-20, 20)):
    sig = inspect.signature(func)
    param_count = len(sig.parameters)
    return [tuple(random.uniform(*input_range) for _ in range(param_count)) for _ in range(n_samples)]
