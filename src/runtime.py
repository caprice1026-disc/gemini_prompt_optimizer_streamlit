from __future__ import annotations


def is_running_with_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        try:
            from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx
        except Exception:
            return False

    try:
        return get_script_run_ctx() is not None
    except Exception:
        return False
