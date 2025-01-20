from document_loader import load_and_split_document

if __name__ == "__main__":
    file_path = "project/Resume Color.pdf"
    chunks = load_and_split_document(file_path)

    if chunks:
        print(f"Loaded {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i + 1}: {chunk}")
    else:
        print("Failed to load document.")
