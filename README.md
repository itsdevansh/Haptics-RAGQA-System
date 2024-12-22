### Project Setup and Execution Guide:

This README file is designed to guide anyone, including those with no prior knowledge of the project, to successfully set up and run the entire project. Follow the detailed steps and procedures below to ensure a smooth experience.

---

### Hardware and Software Requirements:

- **Hardware:**
  - **MacOS:** At least 16GB of RAM and the latest MacOS.
  - **Windows:** Minimum 16GB RAM and the latest Windows.

- **Software:**
  - Python 3.9.6 -> https://www.python.org/downloads/release/python-396/
  - pip -> https://pypi.org/project/pip/
  - venv -> pip install virtualenv

---

### Contributors:

This project was a collaborative effort, and each team member played a crucial role:

- **Devansh Kumar:**  
  - Created the Code file: Chunking_embedding.ipynb
  - Contribution in the RAG_Application.py files is highlighted in the comments in the file.

- **Rasheeq Mohammad:**  
  - Created the Code file: generate_clean_text_files.ipynb
  - Contribution in the RAG_Application.py files is highlighted in the comments in the file.

---

### Project Overview:
This project focuses on setting up a Retrieval-Augmented Generation (RAG) application using the Llama 3.2 model through Ollama. It involves preprocessing research papers, creating a vector database, performing chunking and embedding generation, and integrating a local model alongside OpenAI and Claude APIs to compare the outputs from all three methods. The goal is to demonstrate that, with access to an external knowledge database and effective prompt engineering, a smaller local model can achieve performance comparable to larger LLMs.

---

### Instructions for Setup and Execution:


#### Step 1: Set Up Virtual Environment
- **Create a virtual environment:**  
  ```bash
  python -m venv haptics-rag-project
  ```
- **Activate the virtual environment:**  
  - **MacOS:**  
    ```bash
    cd path/to/haptics-rag-project
    source bin/activate
    cd ..  # Return to the folder with all the code
    ```
  - **Windows:**  
    ```bash
    cd path\to\haptics-rag-project
    Scripts\activate
    cd ..  # Return to the folder with all the code
    ```

#### Step 2: Configure API Keys
- Add the following to your `RAG_Application.py` file where it is indicated in ALL CAPS:  
  - `ANTHROPIC_API_KEY`  
  - `OPENAI_API_KEY`

#### Step 3: Install Requirements
- Run the command:  
  ```bash
  pip install -r requirements.txt
  ```

#### Step 4: Download Data
- [https://uottawa-my.sharepoint.com/personal/rmoha039_uottawa_ca/_layouts/15/guestaccess.aspx?share=EnnJQKXbo3lNlc_Gvxw6En0B0pPaVi7P6l3mynZE0dfZRQ]

> Ensure all files are in the root folder and the virtual environment is activated.

#### Step 5: Download and Install Ollama
- **Download Ollama:** [Download link](https://ollama.com/download/windows)  
- **Setup Guides:**  
  - [Running Llama 3.2 on Windows](https://medium.com/@aleksej.gudkov/how-to-run-llama-3-2-locally-a-complete-guide-36d4a8c7bf94)  
  - [Running Llama 3.2 on Mac](https://www.llama.com/docs/llama-everywhere/running-meta-llama-on-mac)

> **Note:**  
> - After installing Ollama, run the command  `Ollama serve` and then `ollama run llama3.2`. Make sure you donot close the terminal where both ollama is running.  
> - Copy the URL provided by Ollama in the terminal and paste it in the `RAG_Application.py` file as indicated by the comments.

#### Step 6: Execute Project Steps

1. **Run `generate_clean_text_files.ipynb`:**  
   - Process PDFs of research papers to produce cleaned `.txt` files for the next step.  
   - Output folder details are described in the accompanying paper.  

2. **Run `Chunking_embedding.ipynb`:**  
   - Use cleaned text files to create a vector database named `research_papers`.  
   - Specify a chunking strategy (semantic, sentence-based, or fixed).  
   - Generate embeddings of the chunks and store them in the database.

3. **Run `RAG_Application.py`:**  
   - Ensure the URL from Ollama is pasted in the specified location within the file.  
   - This file retrieves context from the vector database, integrates with the local model, and calls APIs (OpenAI and Claude) for comparative results.  
   - If API keys are not provided, comment out the relevant API code.

---

By following these instructions, anyone should be able to successfully set up and run the project. If you encounter any issues or have questions, feel free to contact:  
- **Devansh Kumar:** dkuma079@uottawa.ca  
- **Rasheeq Mohammad:** rmoha039@uottawa.ca  

Thank you for using our project!
