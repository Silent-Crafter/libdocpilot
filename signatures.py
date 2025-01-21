import dspy


class GenerateSearchQuery(dspy.Signature):
    """Extract important keywords from question"""
    question = dspy.InputField(desc="user question")
    keywords = dspy.OutputField()

class GenerateAnswer(dspy.Signature):
    """
Answer question using facts from context only.
If the context is empty or N/A. Always Reply with 'Sorry I cannot assist you with that' regardless of question.
    """
    context = dspy.InputField(desc="will contain relevant facts")
    messages = dspy.InputField(desc="conversation history")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="answer")

class ImageRag(dspy.Signature):
    """
Given the answer, fetch image ids from the context.
The context will have images replaced with unique identifiers like $img-123-123.jpg$
    """
    context = dspy.InputField(desc="will contain relevant facts")
    answer = dspy.InputField(desc="answer")
    image_ids = dspy.OutputField(desc="image ids found in context Example output: $img-123-123.jpg$, $ada-123-123.png$, $asd-133-e9e.jpg$")
