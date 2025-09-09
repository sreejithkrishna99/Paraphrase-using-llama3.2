import ollama
import textwrap

def paraphrase_text(input_text, model_name="llama3.2"):
    """
    Paraphrases long text using Ollama llama3.2 model.
    Handles multiple sentences and large paragraphs.
    """
    prompt = f"""
    You are an AI writing assistant. Paraphrase the following text while:
    - Preserving the original meaning
    - Making it clear, concise, and professional
    - Maintaining natural flow and grammar

    Text to paraphrase:
    {input_text}
    """

    # Stream the response to handle large outputs properly
    response_stream = ollama.generate(
        model=model_name,
        prompt=prompt,
        stream=True  # ✅ Important for handling large texts
    )

    paraphrased_text = ""
    for chunk in response_stream:
        paraphrased_text += chunk.get("response", "")

    return paraphrased_text.strip()

if __name__ == "__main__":
    print("=== Ollama Paraphraser (LLaMA 3.2) ===\n")
    print("Paste multiple sentences or even paragraphs below. Press ENTER twice to finish:\n")

    # Read multi-line input from user
    print("Enter text to paraphrase:\n")
    input_lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        input_lines.append(line)

    input_text = "\n".join(input_lines)

    # Call paraphraser
    result = paraphrase_text(input_text)

    print("\n===== PARAPHRASED TEXT =====\n")
    print(textwrap.fill(result, width=100))
