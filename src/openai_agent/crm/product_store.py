import os

from openai import OpenAI
from dotenv import load_dotenv


def create_the_store():
    """Creates a new vector store"""
    client = OpenAI()

    vector_store = client.vector_stores.create(
        name="MyStore Products"
    )
    print(vector_store)


def add_file_to_openai(file_name: str):
    """Add a file to the OpenAI platform"""
    client = OpenAI()

    response = client.files.create(
        file=open(file_name, 'rb'),
        purpose="assistants"
    )
    print(response)


def add_files_to_store(store_id: str):
    """Add files to the vector store"""
    client = OpenAI()

    response = client.files.list()
    print(response)

    ids = [file.id for file in client.files.list().data]

    for file_id in ids:
        client.vector_stores.files.create(
            vector_store_id=store_id,
            file_id=file_id
        )

def list_files_vector_store(store_id: str):
    """List the files in the vector store"""
    client = OpenAI()

    response = client.vector_stores.files.list(
        vector_store_id=store_id
    )
    print(response)


def list_vector_stores():
    """List the vector stores"""
    client = OpenAI()

    response = client.vector_stores.list()
    print(response)

if __name__ == "__main__":
    _ = load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')

    create_the_store()
    # add_file_to_openai("products/product1.md")
    # add_file_to_openai("products/product2.md")
    # add_file_to_openai("products/product3.md")
    # add_file_to_openai("products/product4.md")
    # add_file_to_openai("products/product5.md")

    vector_store_id = os.getenv('VECTOR_STORE_ID')
    # add_files_to_store(vector_store_id)
    list_files_vector_store(vector_store_id)
    # list_vector_stores()
