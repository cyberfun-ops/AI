import ollama


def generate_answer(question, context):

    prompt = f""" You are a document question-answering assistant. 
    Your job is to answer the user's question using ONLY the information provided in the context. 
    IMPORTANT RULES: 
    1. Do not use outside knowledge. 
    2. Do not make up facts or numbers. 
    3. If the context does not contain enough information, say: "I don't have enough information in the document to answer this." 
    4. Pay close attention to whether a number refers to consolidated or standalone results. 
    5. Cite the page number after each important claim. 
    Use this format: Answer: <your answer> 
    Sources: - Page X 
    - Page Y 
    Context: {context} 
    Question: {question}
    Answer: """

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


if __name__ == "__main__":

    context = """
    The company's revenue declined by 12% during the year.
    The decline was primarily caused by weaker demand in
    the automotive segment and increased pricing pressure.
    """

    question = "Why did revenue decline?"

    answer = generate_answer(question, context)

    #print("\nAnswer:")
    print(answer)