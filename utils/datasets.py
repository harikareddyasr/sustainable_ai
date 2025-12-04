def load_summarization_data():
    # Small toy dataset; we can replace it later with a real one.
    return [
        {
            "id": 1,
            "input": "Artificial intelligence is a field of computer science that focuses on creating intelligent machines.",
            "reference": "AI is a computer science field that builds intelligent machines."
        },
        {
            "id": 2,
            "input": "Climate change is driven largely by human activities that increase greenhouse gas emissions.",
            "reference": "Human-caused greenhouse gases are the main driver of climate change."
        },
    ]


def load_qa_data():
    # Tiny QA subset for testing; later we can plug in SQuAD or another dataset.
    return [
        {
            "id": 1,
            "context": "Barack Obama was the 44th President of the United States.",
            "question": "Who was the 44th President of the United States?",
            "answer": "Barack Obama"
        },
        {
            "id": 2,
            "context": "The capital of France is Paris, which is known for the Eiffel Tower.",
            "question": "What is the capital of France?",
            "answer": "Paris"
        },
    ]
def load_reasoning_data():
    # Simple toy reasoning-style questions with reference answers
    return [
        {
            "id": 1,
            "prompt": "If a laptop uses 50 watts and runs for 2 hours, how much energy does it use in watt-hours?",
            "reference": "100 watt-hours"
        },
        {
            "id": 2,
            "prompt": "Why can quantizing a model to 4-bit sometimes hurt accuracy?",
            "reference": "Because reducing precision can distort the weights and activations, changing what the model has learned."
        },
        {
            "id": 3,
            "prompt": "Explain in one or two sentences why running models on GPUs can be more energy efficient than CPUs.",
            "reference": "GPUs process many operations in parallel, so they can finish the same work faster and sometimes with less total energy."
        },
    ]

