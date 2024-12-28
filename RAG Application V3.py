import gradio as gr
import boto3
import torch
from typing import List, Dict, Union, Set, Optional, Tuple
from langchain import LLMChain, PromptTemplate
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
import logging
from dataclasses import dataclass
import tempfile
from functools import lru_cache
from dotenv import load_dotenv
import asyncio
from concurrent.futures import ThreadPoolExecutor


if torch.backends.mps.is_available():
    device = torch.device("mps")
    print('mps is used.')
else:
    device = 'cpu'
    print('cpu is used.')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
@dataclass
class Config:
    """Configuration class for application settings."""
    PDF_TRACKER_FILE: str = "processed_pdfs.json"
    DATABASE_NAME: str = "haptics_research"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_RESULTS: int = 3
    SUPPORTED_EXTENSIONS: tuple = ('.pptx', '.docx', '.pdf')

class DeviceManager:
    """Manages device selection for torch operations."""
    @staticmethod
    def get_device() -> str:
        if torch.backends.mps.is_available():
            logger.info('Using MPS device')
            return torch.device("mps")
        logger.info('Using CPU device')
        return 'cpu'

class PDFTracker:
    """Manages tracking of processed PDFs."""
    def __init__(self, config: Config):
        self.config = config
        self.processed_pdfs: Set[str] = set()
        self.load_processed_pdfs()

    def load_processed_pdfs(self) -> None:
        """Load the list of previously processed PDFs from JSON file."""
        try:
            if os.path.exists(self.config.PDF_TRACKER_FILE):
                with open(self.config.PDF_TRACKER_FILE, 'r') as f:
                    self.processed_pdfs = set(json.load(f))
        except Exception as e:
            logger.error(f"Error loading PDF tracker: {e}")

    def save_processed_pdfs(self) -> None:
        """Save the updated list of processed PDFs to JSON file."""
        try:
            with open(self.config.PDF_TRACKER_FILE, 'w') as f:
                json.dump(list(self.processed_pdfs), f)
        except Exception as e:
            logger.error(f"Error saving PDF tracker: {e}")

class DatabaseManager:
    """Manages ChromaDB operations."""
    def __init__(self, config: Config):
        self.config = config
        self.client, self.collection = self._initialize_database()

    def _initialize_database(self):
        """Initialize ChromaDB with specified database name."""
        try:
            client = chromadb.PersistentClient(
                path=f"./{self.config.DATABASE_NAME}",
                settings=Settings(
                    allow_reset=True,
                    is_persistent=True
                )
            )
            
            try:
                collection = client.get_collection(name="research_papers")
            except Exception:
                collection = client.create_collection(name="research_papers")
                
            return client, collection
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise

    @staticmethod
    def normalize_for_comparison(text: str) -> str:
        """Normalize text for comparison."""
        return ''.join(char.lower() for char in text if not char.isspace() and char.isalnum())

    def query_chroma(self, query: str) -> List[Dict[str, any]]:
        """Query the Chroma database and return unique most relevant results."""
        results = self.collection.query(
            query_texts=[query],
            n_results=self.config.TOP_K_RESULTS * 5
        )
        
        formatted_results = []
        seen_normalized_texts = set()
        
        for i in range(len(results['ids'][0])):
            current_text = results['documents'][0][i]
            normalized_text = self.normalize_for_comparison(current_text)
            
            if normalized_text not in seen_normalized_texts:
                seen_normalized_texts.add(normalized_text)
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'text': current_text,
                    'distance': results['distances'][0][i]
                })
                
            if len(formatted_results) == self.config.TOP_K_RESULTS:
                break
                
        return formatted_results

class DocumentProcessor:
    """Handles document processing operations."""
    def __init__(self, config: Config):
        self.config = config
        self.llm_client = OllamaLLM(model="llama3.2-vision:latest")
        self.md = MarkItDown(llm_client=self.llm_client)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
        )

    def process_document(self, uploaded_file: Union[str, bytes, object]) -> List[str]:
        """Process uploaded document files and return list of text chunks."""
        file_info = self._get_file_info(uploaded_file)
        
        if not file_info['ext'].endswith(self.config.SUPPORTED_EXTENSIONS):
            raise ValueError(f"Unsupported file type. Supported types are: {self.config.SUPPORTED_EXTENSIONS}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_info['ext']) as temp_file:
            temp_file.write(file_info['content'])
            temp_file_path = temp_file.name

        try:
            logger.info(f"Converting {file_info['name']}...")
            result = self.md.convert(temp_file_path)
            chunks = self.text_splitter.split_text(result.text_content)
            logger.info(f"Successfully converted {file_info['name']}")
            return chunks
        except Exception as e:
            logger.error(f"Error converting {file_info['name']}: {str(e)}")
            raise
        finally:
            os.unlink(temp_file_path)

    @staticmethod
    def _get_file_info(uploaded_file: Union[str, bytes, object]) -> Dict:
        """Extract file information from uploaded file."""
        if isinstance(uploaded_file, str):
            return {
                'name': os.path.basename(uploaded_file),
                'ext': os.path.splitext(uploaded_file)[1].lower(),
                'content': open(uploaded_file, 'rb').read()
            }
        elif isinstance(uploaded_file, bytes):
            return {
                'name': "document.pdf",
                'ext': '.pdf',
                'content': uploaded_file
            }
        else:
            name = getattr(uploaded_file, 'name', 'document.pdf')
            return {
                'name': name,
                'ext': os.path.splitext(name)[1].lower(),
                'content': uploaded_file.read() if hasattr(uploaded_file, 'read') else str(uploaded_file).encode()
            }

class BedrockClient:
    """Handles AWS Bedrock API interactions."""
    def __init__(self):
        self.bedrock_runtime = boto3.client(
            'bedrock-runtime',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name='us-east-1'  # Fixed region format
        )
        self.logger = logging.getLogger(__name__)
    async def invoke_claude(self, message: str, context: str) -> str:
        """Process query using Claude model via AWS Bedrock."""
        try:
            prompt = f"Context: {context}\n\nQuestion: {message}\n\nProvide a clear and concise answer based on the context provided."
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            })

            response = self.bedrock_runtime.invoke_model(
                modelId='anthropic.claude-3-5-sonnet-20240620-v1:0',
                contentType="application/json",
                accept="application/json",
                body=body
            )

            response_body = json.loads(response['body'].read())
            return response_body["content"][0]['text']

        except Exception as e:
            self.logger.error(f"Error in Bedrock Claude processing: {e}")
            return f"An error occurred: {str(e)}"
    
    
    async def invoke_mistral(self, message: str, context: str) -> str:
        """Process query using mistral model via AWS Bedrock."""
        try:
            prompt = f"<s>[INST] Context: {context} Question: {message} [/INST]"

            body = json.dumps({
                "prompt": prompt,
                "max_tokens": 200,
                "temperature": 0.5,
                "top_p": 0.9,
                "top_k": 50
            })

            response = self.bedrock_runtime.invoke_model(
                modelId='mistral.mistral-7b-instruct-v0:2',  # Use appropriate model ID
                contentType="application/json",
                accept="application/json",
                body=body
            )

            response_body = json.loads(response['body'].read())
            return response_body['outputs'][0]['text']

        except Exception as e:
            self.logger.error(f"Error in Bedrock mistral processing: {e}")
            return f"An error occurred: {str(e)}"
        
class HapticsQA:
    def __init__(self):
        self.config = Config()
        self.device = DeviceManager.get_device()
        self.pdf_tracker = PDFTracker(self.config)
        self.db_manager = DatabaseManager(self.config)
        self.doc_processor = DocumentProcessor(self.config)
        self.llm = self._initialize_llm()
        self._setup_chains()
        
        # Initialize Bedrock client
        self.bedrock_client = BedrockClient()
        
        # Initialize additional chains
        self._setup_enhanced_chain()

    def _initialize_llm(self):
        """Initialize the language model."""
        return Ollama(
            model="llama3.2:latest",
            base_url="http://127.0.0.1:11434",
        )
    def _setup_enhanced_chain(self):
        """Set up the enhanced RAG chain."""
        enhance_rag_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a haptics domain researcher instructed to answer the user question in max 3 lines in the tone of a research paper. 
        Your task is to write a small answer to the given question with respect to Haptics. 
        It must be concise 2-line answer that captures the core information need. 
        Focus on the key technical concepts and terminology.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        {query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        
        self.enhance_prompt = PromptTemplate(
            template=enhance_rag_template,
            input_variables=["query"]
        )
        self.enhance_chain = LLMChain(
            prompt=self.enhance_prompt,
            llm=self.llm,
            verbose=False
        )
    async def process_enhanced_query(self, message: str) -> Tuple[str, str]:
        """Process query using enhanced RAG approach."""
        try:
            # Generate reference answer
            llm_generated_reference = self.enhance_chain.run(query=message)
            logger.info(f"Generated reference: {llm_generated_reference}")
            
            # Query database with enhanced prompt
            results = self.db_manager.query_chroma(llm_generated_reference)
            context = self._format_context(results)
            
            # Generate final response
            response = self.standard_chain.run(
                query=message,
                context=context
            )
            cleaned_response = self._clean_response(response)
            
            return context, cleaned_response
        except Exception as e:
            logger.error(f"Error in enhanced query processing: {e}")
            error_message = f"An error occurred: {str(e)}"
            return error_message, error_message

    async def process_mistral_query(self, message: str, context: str) -> str:
        """Process query using mistral model via AWS Bedrock."""
        return await self.bedrock_client.invoke_mistral(message, context)

    async def process_claude_query(self, message: str, context: str) -> str:
        """Process query using Claude model via AWS Bedrock."""
        return await self.bedrock_client.invoke_claude(message, context)
    def handle_pdf_upload(self, pdf_files) -> str:
        """Handle PDF file uploads."""
        try:
            newly_processed = set()
            total_chunks = 0
            skipped_files = 0
            
            for pdf_file in pdf_files:
                filename = Path(pdf_file.name).name
                if filename in self.pdf_tracker.processed_pdfs:
                    skipped_files += 1
                    continue
                    
                chunks = self.doc_processor.process_document(pdf_file)
                self._add_chunks_to_collection(chunks)
                total_chunks += len(chunks)
                newly_processed.add(filename)
            
            self.pdf_tracker.processed_pdfs.update(newly_processed)
            self.pdf_tracker.save_processed_pdfs()
            
            return self._generate_upload_status(len(newly_processed), total_chunks, skipped_files)
        except Exception as e:
            logger.error(f"Error processing PDFs: {str(e)}")
            return f"Error processing PDFs: {str(e)}"

    def _add_chunks_to_collection(self, chunks: List[str]) -> None:
        """Add text chunks to ChromaDB collection."""
        chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]
        self.db_manager.collection.add(
            documents=chunks,
            ids=chunk_ids,
        )

    @staticmethod
    def _generate_upload_status(new_files: int, total_chunks: int, skipped_files: int) -> str:
        """Generate upload status message."""
        status_parts = []
        if new_files:
            status_parts.append(f"Successfully processed {new_files} new PDF(s). Added {total_chunks} text chunks to the database.")
        if skipped_files > 0:
            status_parts.append(f"Skipped {skipped_files} previously processed file(s).")
        return " ".join(status_parts) or "No new files to process."

    def _setup_chains(self):
        """Set up the LLM chains for processing queries."""
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
        
        self.standard_prompt = PromptTemplate(
            template=standard_template, 
            input_variables=["query", "context"]
        )
        self.standard_chain = LLMChain(
            prompt=self.standard_prompt, 
            llm=self.llm, 
            verbose=False
        )

    def _clean_response(self, response: str) -> str:
        """Clean the response to show only the generated answer."""
        if "The question is =" in response:
            response = response.split("The question is =")[1]
            
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
        
        cleaned = "\n".join([line for line in cleaned.split('\n') 
            if not any(x in line.lower() for x in ["date:", "cutting", "today"])])
        
        return cleaned.strip()

    def _format_context(self, results: List[Dict[str, any]]) -> str:
        """Format the ChromaDB results into a string context."""
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(f"{i}. {result['text']}")
        return "\n\n".join(context_parts)

    async def process_query(self, message: str) -> Tuple[str, str]:
        """Process the query and return context and response asynchronously."""
        # Create ThreadPoolExecutor for running synchronous code
        executor = ThreadPoolExecutor(max_workers=3)
        try:
            results = self.db_manager.query_chroma(message)
            context = self._format_context(results)
            
            # Convert synchronous LLMChain.run to async
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.standard_chain.run(
                    query=message,
                    context=context
                )
            )
            cleaned_response = self._clean_response(response)
            
            return context, cleaned_response
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            error_message = f"An error occurred: {str(e)}"
            return error_message, error_message



def create_gradio_interface(qa_system: HapticsQA):
    """Create and configure the Gradio interface with tabs and shared context."""
    with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
        gr.Markdown("# Advanced Haptics QA System")
        gr.Markdown("Ask questions about Haptics and upload PDFs to expand the knowledge base.")
        
        # PDF Upload Section
        with gr.Row():
            pdf_upload = gr.Files(
                label="Upload PDF Documents",
                file_types=[".pdf"],
                file_count="multiple"
            )
        upload_status = gr.Textbox(label="Upload Status", interactive=False)
        
        # Query Input and Shared Context Section
        with gr.Row():
            query_input = gr.Textbox(
                label="Enter your question:",
                placeholder="Type your haptics-related question here...",
                lines=2
            )
            submit_button = gr.Button("Send", variant="primary")
            clear_button = gr.Button("Clear", variant="secondary")
        
        # Shared Context Display
        shared_context = gr.Textbox(
            label="Retrieved Context",
            lines=8,
            interactive=False
        )
        
        # Response Tabs
        with gr.Tabs() as tabs:
            with gr.TabItem("Standard RAG"):
                standard_response = gr.Textbox(
                    label="Standard RAG Response",
                    lines=8,
                    interactive=False
                )
            
            with gr.TabItem("Enhanced RAG"):
                enhanced_response = gr.Textbox(
                    label="Enhanced RAG Response",
                    lines=8,
                    interactive=False
                )
            
            with gr.TabItem("mistral"):
                mistral_response = gr.Textbox(
                    label="mistral Response",
                    lines=8,
                    interactive=False
                )
            
            with gr.TabItem("Claude"):
                claude_response = gr.Textbox(
                    label="Claude Response",
                    lines=8,
                    interactive=False
                )

        async def process_all_queries(message):
            try:
                # Get initial context using standard RAG
                context, standard_result = await qa_system.process_query(message)
                
                # Process with all models using the same context
                enhanced_task = qa_system.process_enhanced_query(message)
                mistral_task = qa_system.process_mistral_query(message, context)
                claude_task = qa_system.process_claude_query(message, context)
                
                # Run remaining tasks concurrently
                enhanced_results, mistral_result, claude_result = await asyncio.gather(
                    enhanced_task,
                    mistral_task,
                    claude_task,
                    return_exceptions=True
                )
                
                # Handle results and exceptions
                def get_result(result):
                    if isinstance(result, Exception):
                        return f"Error: {str(result)}"
                    if isinstance(result, tuple):
                        return result[1]  # Get response part of tuple
                    return str(result)
                
                enhanced_response = get_result(enhanced_results)
                mistral_response = get_result(mistral_result)
                claude_response = get_result(claude_result)
                
                return (
                    context,          # Shared context
                    standard_result,  # Standard RAG response
                    enhanced_response,# Enhanced RAG response
                    mistral_response,     # mistral-4 response
                    claude_response,  # Claude response
                    ""               # Clear input
                )
            except Exception as e:
                error_msg = f"Error processing queries: {str(e)}"
                return (error_msg,) * 5 + ("",)

        # Event Handlers
        pdf_upload.change(
            qa_system.handle_pdf_upload,
            inputs=[pdf_upload],
            outputs=[upload_status]
        )

        # Main query handler
        submit_button.click(
            process_all_queries,
            inputs=[query_input],
            outputs=[
                shared_context,
                standard_response,
                enhanced_response,
                mistral_response,
                claude_response,
                query_input
            ]
        )
        
        # Also trigger on Enter key
        query_input.submit(
            process_all_queries,
            inputs=[query_input],
            outputs=[
                shared_context,
                standard_response,
                enhanced_response,
                mistral_response,
                claude_response,
                query_input
            ]
        )
        
        # Clear button handler
        clear_button.click(
            lambda: "",
            outputs=[query_input]
        )

    return demo
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize the QA system and launch the interface
    qa_system = HapticsQA()
    demo = create_gradio_interface(qa_system)
    demo.queue()  # Enable queuing for async operations
    demo.launch(share=True)