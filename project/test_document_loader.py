from document_loader import load_and_split_document
from qa_system import QASystem

if __name__ == "__main__":
    # File path to the sample document
    file_path = "sample.txt"

    # Load and split the document into chunks
    chunks = load_and_split_document(file_path)
    if not chunks:
        print("Failed to load or split the document.")
        exit()

    print(f"Loaded {len(chunks)} chunks.")

    # Initialize the Q&A system
    qa_system = QASystem()

    # Ask a question
    question = "What is the main topic of the document?"
    print(f"\nQuestion: {question}")

    # Use the first chunk for simplicity
    context = chunks[0].page_content if chunks else ""
    answer = qa_system.answer_question(question, context)

    print(f"Answer: {answer}")
