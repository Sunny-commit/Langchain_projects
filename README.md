# LangChain Projects

A comprehensive collection of LangChain-based projects demonstrating Large Language Model (LLM) integration, Retrieval-Augmented Generation (RAG), and advanced AI applications.

## Overview

This repository contains practical implementations of LangChain, a framework for developing applications powered by language models. Projects include document QA systems, RAG pipelines, chatbots, and intelligent agents.

## Featured Projects

### **Document QA with RAG**
- Load and process documents
- Create vector embeddings
- Semantic search and retrieval
- Context-aware question answering
- Support for multiple document formats

### **RAG Systems**
- Retrieval-Augmented Generation
- Vector database integration
- Context management
- Multi-document handling

### **Chatbots & Conversational AI**
- Multi-turn conversations
- Memory management
- Context preservation
- Response generation

### **Intelligent Agents**
- Tool integration
- Decision-making logic
- Multi-step reasoning
- Error handling and fallbacks

## Technology Stack

### Core Framework
- **LangChain**: LLM application framework
- **Python 3.8+**: Programming language

### LLM Providers
- **OpenAI**: GPT-3.5, GPT-4
- **Anthropic**: Claude models
- **Google**: PaLM, Vertex AI
- **HuggingFace**: Open-source models

### Vector Databases
- **Pinecone**: Cloud vector database
- **Weaviate**: Open-source vector DB
- **Milvus**: Scalable vector DB
- **Chroma**: Local vector store
- **FAISS**: Facebook similarity search

### Supporting Libraries
- **Pandas**: Data handling
- **NumPy**: Numerical computing
- **Requests**: HTTP library
- **PyPDF2**: PDF processing
- **LangSmith**: LangChain monitoring

## Project Structure

```
Langchain_projects/
├── Document QA with RAG.py        # Main RAG implementation
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variables template
├── notebooks/                      # Jupyter notebooks
│   ├── rag_tutorial.ipynb
│   ├── chatbot_example.ipynb
│   └── agent_examples.ipynb
├── data/                           # Sample documents
│   ├── documents/
│   ├── pdfs/
│   └── texts/
├── utils/                          # Helper functions
│   ├── document_loader.py
│   ├── embeddings.py
│   └── vector_store.py
└── README.md
```

## Installation & Setup

### Prerequisites
```bash
- Python 3.8+
- pip or conda
- API keys (OpenAI, HuggingFace, etc.)
```

### Installation Steps

1. **Clone Repository**
```bash
git clone https://github.com/Sunny-commit/Langchain_projects.git
cd Langchain_projects
```

2. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install langchain openai python-dotenv pandas numpy requests PyPDF2
pip install -r requirements.txt
```

4. **Setup Environment Variables**
```bash
cp .env.example .env
# Edit .env and add your API keys
OPENAI_API_KEY=your_key_here
HUGGINGFACE_API_KEY=your_key_here
```

5. **Run Example**
```bash
python "Document QA with RAG.py"
```

## Document QA with RAG Implementation

### Basic Workflow

```python
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# 1. Load documents
loader = PyPDFLoader("document.pdf")
documents = loader.load()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
chunks = splitter.split_documents(documents)

# 3. Create embeddings
embeddings = OpenAIEmbeddings()

# 4. Create vector store
vector_store = FAISS.from_documents(chunks, embeddings)

# 5. Create QA chain
llm = ChatOpenAI(model_name="gpt-3.5-turbo")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever()
)

# 6. Ask questions
result = qa_chain.run("What is the main topic?")
print(result)
```

## Key Concepts

### Retrieval-Augmented Generation (RAG)
```
Query
  ↓
Embedding
  ↓
Vector Search
  ↓
Document Retrieval
  ↓
Context Building
  ↓
LLM Processing
  ↓
Response Generation
```

### Memory Types
- **Buffer Memory**: Last N interactions
- **Summary Memory**: Summarized history
- **Conversation Buffer**: Full history
- **Entity Memory**: Tracked entities

### Chain Types
- **stuff**: Direct context passing
- **map_reduce**: Process chunks separately
- **refine**: Iterative refinement
- **map_rerank**: Ranked selection

## Features & Capabilities

### Document Processing
- ✅ PDF extraction and parsing
- ✅ Text chunking strategies
- ✅ Metadata preservation
- ✅ Document filtering

### Semantic Search
- ✅ Vector similarity search
- ✅ Hybrid search (metadata + content)
- ✅ Relevance scoring
- ✅ Result ranking

### LLM Integration
- ✅ Multiple LLM providers
- ✅ Model switching
- ✅ Temperature control
- ✅ Token limits management

### Conversation Management
- ✅ Multi-turn dialogue
- ✅ Context preservation
- ✅ Memory management
- ✅ Session handling

## Advanced Features

### Custom Tools & Agents
```python
from langchain.agents import Tool, initialize_agent
from langchain.agents import AgentType

tools = [
    Tool(
        name="Document Search",
        func=search_documents,
        description="Search in documents"
    ),
    Tool(
        name="Calculator",
        func=calculate,
        description="Perform calculations"
    )
]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)
```

### Custom Prompts
```python
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["question", "context"],
    template="""
    Answer based on context:
    Context: {context}
    Question: {question}
    """
)
```

### Chain Composition
```python
from langchain.chains import SequentialChain

chain1 = chain_type_1
chain2 = chain_type_2

overall_chain = SequentialChain(
    chains=[chain1, chain2],
    verbose=True
)
```

## Configuration Options

### Embeddings Configuration
```python
embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key=api_key
)
```

### LLM Configuration
```python
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.7,
    max_tokens=500,
    top_p=0.9
)
```

### Vector Store Configuration
```python
vector_store = FAISS.from_documents(
    documents,
    embeddings,
    metadatas=[...]
)
```

## Best Practices

✅ **Chunk Management**
- Optimize chunk size (500-2000 tokens)
- Use appropriate overlap (10-20%)
- Preserve document structure

✅ **Prompt Engineering**
- Clear instructions
- Relevant examples
- Output format specification

✅ **Token Management**
- Monitor token usage
- Implement token limits
- Cache repeated calls

✅ **Error Handling**
- API error handling
- Fallback mechanisms
- Retry logic

✅ **Performance**
- Use async operations
- Batch processing
- Caching strategies

## Common Use Cases

### Customer Support
- Document-based help systems
- FAQ automation
- Ticketing assistance

### Research Assistant
- Paper summarization
- Information extraction
- Citation management

### Code Documentation
- Codebase Q&A
- API documentation
- Tutorial generation

### Content Generation
- Blog post writing
- Report generation
- Summary creation

## Troubleshooting

### API Rate Limits
```python
from time import sleep

# Add delays between requests
for query in queries:
    result = qa_chain.run(query)
    sleep(1)
```

### Token Limits
```python
# Use shorter context window
retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}  # Fewer documents
)
```

### Memory Issues
```python
# Use summary memory for long conversations
from langchain.memory import ConversationSummaryMemory

memory = ConversationSummaryMemory(llm=llm)
```

## Performance Optimization

- **Caching Embeddings**: Reuse computed embeddings
- **Batch Processing**: Process multiple documents
- **Asynchronous Calls**: Parallel API requests
- **Vector Store Indexing**: Fast similarity search

## Deployment Considerations

### Production Checklist
- [ ] API key management (environment variables)
- [ ] Rate limiting implementation
- [ ] Error handling and logging
- [ ] Monitoring and alerting
- [ ] Cost tracking
- [ ] Model versioning
- [ ] Load balancing

### Scaling Strategies
- Distributed vector stores
- Load balancing across instances
- Caching layers
- Async processing

## Integration Examples

### FastAPI Integration
```python
from fastapi import FastAPI

app = FastAPI()

@app.post("/ask")
async def ask_question(question: str):
    result = qa_chain.run(question)
    return {"answer": result}
```

### Streamlit Integration
```python
import streamlit as st

st.title("Document QA System")
question = st.text_input("Ask a question:")
if question:
    answer = qa_chain.run(question)
    st.write(answer)
```

## Learning Resources

### Documentation
- [LangChain Official Docs](https://docs.langchain.com)
- [LangChain API Reference](https://api.python.langchain.com)
- [LangChain GitHub](https://github.com/langchain-ai/langchain)

### Tutorials
- Vector database setup
- Custom chain creation
- Agent development
- Tool integration

### Examples
- QA systems
- Chatbots
- Agents
- Chains

## Contributing

1. Fork the repository
2. Create feature branch
3. Add implementation
4. Include documentation
5. Submit pull request

## Project Statistics

- **Total Projects**: Multiple implementations
- **Lines of Code**: 500+
- **Supported LLMs**: 5+
- **Vector Stores**: 4+
- **Documentation**: Comprehensive

## Future Enhancements

- [ ] Multi-modal RAG (images, videos)
- [ ] Real-time streaming responses
- [ ] Advanced memory management
- [ ] Graph-based reasoning
- [ ] Custom tool library
- [ ] Performance benchmarks
- [ ] Web UI dashboard

## License

MIT License - Free to use for personal and commercial projects

## Author

Pateti Chandu (Sunny-commit)

## Support & Contact

- **GitHub Issues**: Report bugs and suggest features
- **Discussions**: Share ideas and learnings
- **Email**: Contact for collaboration

## Acknowledgments

- LangChain team for the framework
- OpenAI and other LLM providers
- Vector database communities

## Citation

```
@repository{Langchain_Projects,
  title={LangChain Projects Collection},
  author={Pateti Chandu},
  year={2025},
  url={https://github.com/Sunny-commit/Langchain_projects}
}
```
