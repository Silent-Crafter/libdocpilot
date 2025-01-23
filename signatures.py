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
Given the mappings of image file names as keys and their description as values, place the image files at appropriate locations as html.
    """
    ground_truth = dspy.InputField(desc="image file name mapping with their description")
    question = dspy.InputField(desc="context")
    response = dspy.OutputField(desc="image files paired with context at appropriate locations")
