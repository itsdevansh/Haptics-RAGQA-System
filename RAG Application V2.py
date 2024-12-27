import gradio as gr
import torch
from typing import List, Dict, Union, Set
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain import LLMChain, PromptTemplate
from langchain_huggingface import HuggingFacePipeline
import chromadb
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE
import json
import os
from pathlib import Path
from openai import OpenAI
import anthropic
from gtts import gTTS
import tempfile
import pygame
from pathlib import Path
from langchain.llms import Ollama
from langchain_ollama import OllamaLLM
from markitdown import MarkItDown
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile

# Device setup if MacOS
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print('mps is used.')
else:
    device = 'cpu'
    print('cpu is used.')

# Constants for file and database names
PDF_TRACKER_FILE = "processed_pdfs.json"
DATABASE_NAME = "haptics_research"

def load_processed_pdfs() -> Set[str]:
    """
    Load the list of previously processed PDFs from JSON file.
    Returns a set of processed PDF filenames.
    """
    try:
        if os.path.exists(PDF_TRACKER_FILE):
            with open(PDF_TRACKER_FILE, 'r') as f:
                return set(json.load(f))
        return set()
    except Exception as e:
        print(f"Error loading PDF tracker: {e}")
        return set()

def save_processed_pdfs(processed_pdfs: Set[str]) -> None:
    """
    Save the updated list of processed PDFs to JSON file.
    """
    try:
        with open(PDF_TRACKER_FILE, 'w') as f:
            json.dump(list(processed_pdfs), f)
    except Exception as e:
        print(f"Error saving PDF tracker: {e}")

def initialize_database():
    """
    Initialize ChromaDB with the specified database name.
    Creates the database if it doesn't exist.
    """
    try:
        client = chromadb.PersistentClient(
            path=f"./{DATABASE_NAME}",  # Specify the database path
            settings=Settings(
                allow_reset=True,
                is_persistent=True
            ),
            tenant=DEFAULT_TENANT,
        )
        
        # Try to get or create the collection
        try:
            collection = client.get_collection(name="research_papers")
        except Exception:
            collection = client.create_collection(name="research_papers")
            
        return client, collection
    except Exception as e:
        print(f"Error initializing database: {e}")
        raise


llm = Ollama(
    model="llama3.2:latest",  # Use the exact model name from `ollama list`
    base_url="http://127.0.0.1:11434",  # Default Ollama server URL ######PASTE YOUR OLLAMA URL 
)

from typing import List
import os
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter

def process_pdf(uploaded_file: Union[str, bytes, object]) -> List[str]:
    """
    Process uploaded document files (PDF, PPTX, DOCX) and return list of text chunks.
    
    Args:
        uploaded_file: Can be a file path (str), bytes content, or file-like object
        
    Returns:
        List[str]: List of text chunks from the document
    """
    # Define supported file extensions
    supported_extensions = ('.pptx', '.docx', '.pdf')
    
    # Handle different input types
    if isinstance(uploaded_file, str):
        # If it's a file path
        file_ext = os.path.splitext(uploaded_file)[1].lower()
        file_name = os.path.basename(uploaded_file)
        file_content = open(uploaded_file, 'rb').read()
    elif isinstance(uploaded_file, bytes):
        # If it's bytes content
        file_name = "document.pdf"  # Default name
        file_ext = '.pdf'  # Default extension
        file_content = uploaded_file
    else:
        # If it's a file-like object
        file_name = getattr(uploaded_file, 'name', 'document.pdf')
        file_ext = os.path.splitext(file_name)[1].lower()
        try:
            file_content = uploaded_file.read()
        except AttributeError:
            file_content = str(uploaded_file).encode()

    # Validate file extension
    if not file_ext.endswith(supported_extensions):
        raise ValueError(f"Unsupported file type. Supported types are: {supported_extensions}")

    # Initialize MarkItDown with Ollama client    
    client = OllamaLLM(model="llama3.2-vision:latest")
    md = MarkItDown(llm_client=client)

    # Save content to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
        temp_file.write(file_content)
        temp_file_path = temp_file.name

    try:
        # Convert document to markdown using MarkItDown
        print(f"\nConverting {file_name}...")
        result = md.convert(temp_file_path)
        text = result.text_content

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_text(text)
        
        print(f"Successfully converted {file_name}")
        return chunks

    except Exception as e:
        print(f"Error converting {file_name}: {str(e)}")
        raise

    finally:
        # Clean up temporary file
        os.unlink(temp_file_path)


def add_chunks_to_collection(chunks: List[str], collection: chromadb.Collection) -> None:
    """
    Add text chunks to ChromaDB collection.
    """
    # Generate IDs for chunks
    chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]
    
    # Add chunks to collection
    collection.add(
        documents=chunks,
        ids=chunk_ids,
    )



def normalize_for_comparison(text: str) -> str:
    """
    Normalize text for comparison by removing all whitespace and punctuation.
    """
    # Remove all whitespace, punctuation, and convert to lowercase
    return ''.join(char.lower() for char in text if not char.isspace() and char.isalnum())

def query_chroma(query: str, collection: chromadb.Collection, top_k: int = 3) -> List[Dict[str, any]]:
    """
    Query the Chroma database and return top_k in this case 3, unique most relevant results.
    Handles cases where texts might be formatted differently but contain the same content.
    Specifically handles whitespace variations and minor formatting differences.
    """
    # Query with more results than needed to account for duplicates
    results = collection.query(
        query_texts=[query],
        n_results=top_k * 5  # Request more results to ensure enough unique entries after filtering
    )
    
    # Create a list to track processed results
    formatted_results = []
    # Track normalized versions of added texts
    seen_normalized_texts = set()
    
    # Process results and remove similar entries
    for i in range(len(results['ids'][0])):
        current_text = results['documents'][0][i]
        normalized_text = normalize_for_comparison(current_text)
        
        # Skip if we've seen this normalized text before
        if normalized_text not in seen_normalized_texts:
            seen_normalized_texts.add(normalized_text)
            formatted_results.append({
                'id': results['ids'][0][i],
                'text': current_text,  # Keep original formatting for display
                'distance': results['distances'][0][i]
            })
            
        # Break if we have enough unique results
        if len(formatted_results) == top_k:
            break
            
    return formatted_results

def format_context(results: List[Dict[str, any]]) -> str:
    """Format the ChromaDB results into a string context."""
    context_parts = []
    for i, result in enumerate(results, 1):
        context_parts.append(f"{i}. {result['text']}")
    return "\n\n".join(context_parts)


standard_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert AI assistant specializing in haptics domain, tasked with providing accurate, concise responses.
Explain concepts using the context like i am a 15 year old.
Offer clear, direct explanations of key haptic concepts.
You are instructed to use the context given below to answer the user question.
If the context is not of good quality then provide a general answer to the user question.
The context is  you need to derive your answer from = 
{context}
The question is =
<|eot_id|><|start_header_id|>user<|end_header_id|>
{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>Your Answer = """

standard_prompt = PromptTemplate(template=standard_template, input_variables=["query", "context"])
standard_chain = LLMChain(prompt=standard_prompt, llm=llm, verbose=False)

# Initialize ChromaDB client
# client = chromadb.PersistentClient(
#     settings=Settings(),
#     tenant=DEFAULT_TENANT,
#     database=DEFAULT_DATABASE
# )
# collection = client.get_collection(name="research_papers")

def clean_response(response: str) -> str:
    """Clean the response to show only the generated answer."""
    # First find everything after "The question is ="
    if "The question is =" in response:
        response = response.split("The question is =")[1]
        
    # Remove any remaining system markers
    markers = [
        "<|begin_of_text|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
        "system",
        "user",
        "assistant",
        "Your Answer ="
    ]
    
    cleaned = response
    for marker in markers:
        cleaned = cleaned.replace(marker, "")
    
    # Remove any lines containing unwanted phrases
    cleaned = "\n".join([line for line in cleaned.split('\n') 
    if not any(x in line.lower() for x in ["date:", "cutting", "today"])])
    
    # Final cleanup
    cleaned = cleaned.strip()
    
    return cleaned


def enchanced_rag(message: str) -> str:
    """Enhanced RAG approach: LLM generating small rewriting before vector search."""

    enhance_rag_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a haptics domain researcher instructed to answer the user question in max 3 lines in the tone of a research paper. Your task is to write a small answer to the given question with respect to Haptics. It must be concise 2-line answer that captures the core information need. Focus on the key technical concepts and terminology.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    {query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    enhance_rag_prompt = PromptTemplate(template=enhance_rag_template, input_variables=["query"])
    enhance_rag_chain = LLMChain(prompt=enhance_rag_prompt, llm=llm, verbose=False)

    try:
        llm_generated_reference_answer = enhance_rag_chain.run(query=message)
        print(f"llm_generated_reference_answer: {llm_generated_reference_answer}")
        
        results = query_chroma(llm_generated_reference_answer, collection)
        context = format_context(results)
        
        response = standard_chain.run(query=message, context=context)
        return clean_response(response)
    except Exception as e:
        return f"An error occurred: {str(e)}"




def process_query(message: str) -> tuple[str, str, str, str]:
    """Process the query and return context and responses from all models."""
    try:
        # Get context and RAG response
        results = query_chroma(message, collection)
        context = format_context(results)
        rag_response = standard_chain.run(query=message, context=context)
        # enchanced_rag_reponse = enchanced_rag(message)   #this is the enhanced rag approach, we didn't implement it in the user study prototype as its resource intensive
        cleaned_rag_response = clean_response(rag_response)

        return context, cleaned_rag_response
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        return error_message, error_message, error_message, error_message

def process_query_and_clear(message: str) -> tuple[str, str, str, str, str]:
    """Process the query and return context, responses from all models, and empty string to clear input."""
    context, rag_response = process_query(message)
    return context, rag_response, ""
def handle_pdf_upload(pdf_files) -> str:
    """
    Handle uploaded PDF files and add their content to the database.
    Checks if files were previously processed and only processes new ones.
    Returns a status message.
    """
    try:
        # Load the set of previously processed PDFs
        processed_pdfs = load_processed_pdfs()
        
        # Track new files processed in this session
        newly_processed = set()
        total_chunks = 0
        skipped_files = 0
        
        for pdf_file in pdf_files:
            # Get the filename and check if it was previously processed
            filename = Path(pdf_file.name).name
            if filename in processed_pdfs:
                skipped_files += 1
                continue
                
            chunks = process_pdf(pdf_file)
            add_chunks_to_collection(chunks, collection)
            total_chunks += len(chunks)
            newly_processed.add(filename)
        
        # Update and save the processed PDFs list
        processed_pdfs.update(newly_processed)
        save_processed_pdfs(processed_pdfs)
        
        # Prepare status message
        status_parts = []
        if newly_processed:
            status_parts.append(f"Successfully processed {len(newly_processed)} new PDF(s). Added {total_chunks} text chunks to the database.")
        if skipped_files > 0:
            status_parts.append(f"Skipped {skipped_files} previously processed file(s).")
        
        return " ".join(status_parts) or "No new files to process."
    except Exception as e:
        return f"Error processing PDFs: {str(e)}"

# Initialize the database and collection at startup
client, collection = initialize_database()



"""
Following section is the creating of web UI using gradio library
"""
with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# Llama 3.2 Haptics QA System")
    gr.Markdown("Ask questions about Haptics and upload PDFs to expand the knowledge base.")
    
    with gr.Row():
        pdf_upload = gr.Files(
            label="Upload PDF Documents",
            file_types=[".pdf"],
            file_count="multiple"
        )
    
    upload_status = gr.Textbox(
        label="Upload Status",
        interactive=False
    )
    
    with gr.Row():
        input_text = gr.Textbox(
            label="Enter your question:",
            placeholder="Type your haptics-related question here...",
            lines=2
        )
        submit_btn = gr.Button("Send", variant="primary", size="sm")
        clear_btn = gr.Button("Clear", variant="secondary", size="sm")

    with gr.Row():
        with gr.Column():
            with gr.Group():
                context_box = gr.Textbox(
                    label="Retrieved Context",
                    lines=20,
                    interactive=False,
                    visible=True,
                    max_lines=20,
                )
        
        with gr.Column():
            with gr.Group():
                rag_response_box = gr.Textbox(
                    label="RAG Response",
                    lines=20,
                    interactive=False,
                    visible=True,
                    max_lines=20,
                )
    
    # Event handlers
    pdf_upload.change(
        handle_pdf_upload,
        inputs=[pdf_upload],
        outputs=[upload_status]
    )
    
    input_text.submit(
        process_query_and_clear,
        inputs=[input_text],
        outputs=[context_box, rag_response_box, input_text]
    )

    submit_btn.click(
        process_query_and_clear,
        inputs=[input_text],
        outputs=[context_box, rag_response_box, input_text]
    )
    
    clear_btn.click(lambda: "", inputs=[], outputs=[input_text])

if __name__ == "__main__":
    demo.launch(share=True)