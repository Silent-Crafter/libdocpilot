import dspy


class GenerateSearchQuery(dspy.Signature):
    """Generate a relevent search query from user question"""
    context = dspy.InputField(desc="will contain relevent facts")
    question = dspy.InputField(desc="user question")
    query = dspy.OutputField()

class GenerateAnswer(dspy.Signature):
    """Answer questions with factoid answers"""
    context = dspy.InputField(desc="will contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="based on the requirement of question, answer in 1 to 5 words or in a paragraph or in a tabular format of csv")

