from typing import Any

class QueryPrompt:
    """Generate search query according to the provided question and past context. Respond ONLY with 4-5 precise comma seperated keywords and nothing else"""
    past_context = None
    question = None

class AnswerPrompt:
    """Answer question using facts from context and previous conversations only.\nIf there are past messages, refer to them when answering users's query with respect to context.\nIf the context is empty or N/A and users's current questions doesn't align with past messages, always Reply with 'Sorry I cannot assist you with that'."""
    context = None
    messages = None
    question = None

def build_prompt_from_signature(signature, inputs: dict[str, Any]):
    doc = signature.__doc__ or ""
    
    fields = {
        k: v 
        for k, v in vars(signature).items()
        if not k.startswith("__") and not callable(v)
    }

    prompt = doc + "\n\n"
    for field in fields:
        value = inputs.pop(field, "")
        prompt += f"[[ {field.upper()} ]]\n{value}\n\n"

    return prompt

