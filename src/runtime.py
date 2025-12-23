from __future__ import annotations


def is_running_with_streamlit() -> bool:
    import logging

    logger_names = [
        "streamlit.runtime.scriptrunner_utils.script_run_context",
        "streamlit.runtime.scriptrunner.script_run_context",
    ]
    prev_levels = {}
    for name in logger_names:
        logger = logging.getLogger(name)
        prev_levels[name] = logger.level
        logger.setLevel(logging.ERROR)

    try:
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
    finally:
        for name, level in prev_levels.items():
            logging.getLogger(name).setLevel(level)
