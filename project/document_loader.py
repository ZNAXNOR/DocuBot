from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter


def load_and_split_document(file_path: str):
    """
    Loads a document from the given file path and splits it into smaller chunks.

    Args:
        file_path (str): Path to the document file.

    Returns:
        List[str]: List of document chunks.
    """
    try:
        # Load the document
        loader = TextLoader(file_path)
        document = loader.load()

        # Split the document into smaller chunks for easier processing
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=100,
        )
        document_chunks = text_splitter.split_documents(document)

        return document_chunks
    except Exception as e:
        print(f"Error loading document: {e}")
        return None
