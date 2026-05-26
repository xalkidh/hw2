import uvicorn
from src.rag import build_vector_store, VECTOR_STORE_PATH
import os

if __name__ == "__main__":
    # Χτίζουμε το vector store αν δεν υπάρχει ήδη
    if not os.path.exists(VECTOR_STORE_PATH) or not os.listdir(VECTOR_STORE_PATH):
        print("Building vector store...")
        build_vector_store()
        print("Vector store ready!")
    else:
        print("Vector store already exists, skipping build.")

    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)