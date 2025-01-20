import dspy


class GenerateSearchQuery(dspy.Signature):
    """Extract important keywords from question"""
    question = dspy.InputField(desc="user question")
    keywords = dspy.OutputField()

class GenerateAnswer(dspy.Signature):
    """Answer question using facts from context only. If the context is empty or N/A. Always Reply with 'Sorry I cannot assist you with that' regardless of question. Use image name text enclosed in $ when asked"""
    context = dspy.InputField(desc="will contain relevant facts")
    messages = dspy.InputField(desc="conversation history")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="based on the requirement of question, answer in 1 to 5 words or in a paragraph")
    images = dspy.OutputField(desc="image file names from context (enclosed in $) relevant to answer as csv")

