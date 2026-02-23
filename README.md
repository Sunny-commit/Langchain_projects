# 🔗 LangChain Projects - Large Language Model Applications

A **comprehensive guide to LangChain** for building intelligent applications with LLMs, implementing RAG (Retrieval-Augmented Generation), agents, and memory management.

## 🎯 Overview

This project covers:
- ✅ LLM integration (OpenAI, Hugging Face)
- ✅ Prompt engineering & templates
- ✅ Chains & sequential operations
- ✅ RAG (Retrieval-Augmented Generation)
- ✅ Agents & tools
- ✅ Memory management
- ✅ Document processing

## 🧠 Basic LLM Integration

```python
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StreamingStdOutCallbackHandler

# Initialize LLM
llm = OpenAI(
    temperature=0.7,
    max_tokens=512,
    openai_api_key="your-api-key"
)

# Chat model
chat = ChatOpenAI(
    temperature=0.7,
    model_name="gpt-3.5-turbo"
)

# Simple query
response = llm.predict(text="What is machine learning?")
print(response)

# Streaming output
chat_with_streaming = ChatOpenAI(
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)
```

## 📝 Prompt Engineering

```python
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate

# Basic prompt template
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} in simple terms for beginners."
)

# Chat prompt template
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert Python programmer."),
    ("user", "Explain the {concept}")
])

# Few-shot prompting
examples = [
    {
        "input": "What is inheritance?",
        "output": "Inheritance is when a class inherits properties from another class."
    },
    {
        "input": "What is polymorphism?",
        "output": "Polymorphism allows objects to take on multiple forms."
    }
]

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=PromptTemplate(
        input_variables=["input", "output"],
        template="Q: {input}\nA: {output}"
    ),
    prefix="Answer programming questions:",
    input_variables=["question"],
    example_separator="\n\n"
)
```

## ⛓️ Chains

```python
from langchain.chains import LLMChain, SequentialChain, SimpleSequentialChain
from langchain.chains.retrieval_qa.base import RetrievalQA

# Single chain
analysis_prompt = PromptTemplate(
    input_variables=["article"],
    template="Summarize this article:\n{article}"
)

chain = LLMChain(llm=llm, prompt=analysis_prompt)
result = chain.run(article="...")

# Sequential chain (multiple steps)
# Step 1: Generate questions
question_prompt = PromptTemplate(
    input_variables=["context"],
    template="Generate 5 interview questions about:\n{context}"
)
question_chain = LLMChain(llm=llm, prompt=question_prompt)

# Step 2: Answer questions
answer_prompt = PromptTemplate(
    input_variables=["questions"],
    template="Answer these questions:\n{questions}"
)
answer_chain = LLMChain(llm=llm, prompt=answer_prompt)

# Combine chains
overall_chain = SequentialChain(
    chains=[question_chain, answer_chain],
    output_variables=["text"],
    verbose=True
)

# Run chain with context
result = overall_chain(context="Machine Learning fundamentals")
```

## 📚 RAG (Retrieval-Augmented Generation)

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

class RAGPipeline:
    """Retrieval-Augmented Generation"""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = None
        self.qa_chain = None
    
    def load_documents(self, file_path):
        """Load and split documents"""
        if file_path.endswith('.pdf'):
            loader = PDFLoader(file_path)
        else:
            loader = TextLoader(file_path)
        
        documents = loader.load()
        
        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        chunks = splitter.split_documents(documents)
        return chunks
    
    def create_vectorstore(self, documents):
        """Create vector embeddings"""
        self.vectorstore = FAISS.from_documents(
            documents,
            self.embeddings
        )
    
    def setup_qa_chain(self, llm):
        """Create QA chain"""
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(),
            return_source_documents=True
        )
    
    def query(self, question):
        """Query documents"""
        result = self.qa_chain({
            "query": question
        })
        
        return {
            "answer": result["result"],
            "sources": result.get("source_documents", [])
        }

# Usage
rag = RAGPipeline()
documents = rag.load_documents("document.pdf")
rag.create_vectorstore(documents)
rag.setup_qa_chain(llm)

answer = rag.query("What is the main topic?")
print(answer)
```

## 🤖 Agents & Tools

```python
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.tools import DuckDuckGoSearchRun

class CustomTools:
    """Define custom tools for agents"""
    
    @staticmethod
    def create_tools():
        """Create available tools"""
        search = DuckDuckGoSearchRun()
        
        tools = [
            Tool(
                name="Search",
                func=search.run,
                description="Useful for answering questions about current events"
            ),
            Tool(
                name="Calculator",
                func=lambda x: str(eval(x)),
                description="Useful for math calculations"
            ),
            Tool(
                name="Python REPL",
                func=lambda x: eval(x),
                description="Useful for executing Python code"
            )
        ]
        
        return tools

# Create agent
tools = CustomTools.create_tools()

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    max_iterations=5
)

# Run agent
result = agent.run("What's the current Bitcoin price? Multiply by 2.")
```

## 💾 Memory Management

```python
from langchain.memory import ConversationMemory, ConversationBufferMemory
from langchain.chains import ConversationChain

# Conversation memory
memory = ConversationBufferMemory()

# Conversation chain with memory
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# Multi-turn conversation
print(conversation.run(input="Hi, my name is Alice"))
print(conversation.run(input="What's my name?"))  # Remembers context
print(conversation.run(input="Tell me about Python"))
```

## 🔍 Document Processing

```python
from langchain.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter

class DocumentProcessor:
    """Process various document formats"""
    
    @staticmethod
    def load_directory(directory_path, file_type="*.txt"):
        """Load all documents from directory"""
        loader = DirectoryLoader(
            directory_path,
            glob=file_type,
            loader_cls=UnstructuredFileLoader
        )
        
        documents = loader.load()
        return documents
    
    @staticmethod
    def smart_chunk(documents, chunk_size=1000, overlap=200):
        """Intelligent document chunking"""
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=overlap
        )
        
        chunks = splitter.split_documents(documents)
        return chunks
```

## 💡 Interview Talking Points

**Q: When use RAG vs fine-tuning?**
```
Answer:
- RAG: Fast, uses existing models, good for knowledge updates
- Fine-tuning: Slow, learn domain-specific patterns, permanent knowledge
- RAG better for: Up-to-date info, custom documents
- Fine-tuning better for: Style, rare concepts, performance
```

**Q: Design effective prompts?**
```
Answer:
- Clear role/context for model
- Few-shot examples
- Specific output format
- Chain of thought
- Temperature tuning (0=deterministic, 1=creative)
```

## 🌟 Portfolio Value

✅ LLM integration
✅ Prompt engineering
✅ Chain architecture
✅ RAG implementation
✅ Agent systems
✅ Memory management
✅ Document processing

---

**Technologies**: LangChain, OpenAI, FAISS, Embeddings

