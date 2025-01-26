from document_loader import load_and_split_document
from qa_system import QASystem

if __name__ == "__main__":
    file_path = "sample.txt"
    chunks = load_and_split_document(file_path)

    if not chunks:
        print("Failed to load or split the document.")
        exit()

    print(f"Loaded {len(chunks)} chunks.")

    qa_system = QASystem()

    question = "Explain topic"
    print(f"Question: {question}")

    # Dynamically build context from chunks
    context = " "
    for chunk in chunks:
        if len(context.split()) + len(chunk.page_content.split()) <= 512:  # Token limit example
            context += f" {chunk.page_content}"
        else:
            break

    answer = qa_system.answer_question(question, context)
    print(f"Answer: {answer}")
