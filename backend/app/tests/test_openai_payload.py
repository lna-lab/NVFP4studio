from app.api.routes.openai import _prepare_payload


def test_prepare_payload_disables_thinking_by_default():
    payload = _prepare_payload({"model": "demo-model"})

    assert payload["chat_template_kwargs"]["enable_thinking"] is False


def test_prepare_payload_preserves_existing_chat_template_kwargs():
    payload = _prepare_payload(
        {
            "model": "demo-model",
            "chat_template_kwargs": {"custom_flag": True},
        }
    )

    assert payload["chat_template_kwargs"]["enable_thinking"] is False
    assert payload["chat_template_kwargs"]["custom_flag"] is True


def test_prepare_payload_respects_explicit_enable_thinking():
    payload = _prepare_payload(
        {
            "model": "demo-model",
            "chat_template_kwargs": {"enable_thinking": True},
        }
    )

    assert payload["chat_template_kwargs"]["enable_thinking"] is True
