# LangGraph Agent HR Assistant

A sophisticated HR assistant powered by LangChain, LangGraph, and multiple LLM models. This project demonstrates the implementation of a conversational AI system that can help with HR-related queries and team building recommendations.

## Project Overview

This project implements an intelligent HR assistant that can:
- Process natural language queries about employees and team building
- Access and search through employee records
- Provide recommendations for team composition
- Maintain conversation context across multiple interactions

## Technology Stack

### Core Technologies
- **TypeScript/Node.js** - Main development language and runtime
- **Express.js** - Web server framework
- **MongoDB Atlas** - Database with vector search capabilities
- **LangChain** - LLM framework for building AI applications
- **LangGraph** - Framework for building stateful AI agent workflows

### LLM Models Used
1. **Claude 3 Sonnet (Anthropic)** - Primary model for agent reasoning and responses
2. **Mistral 7B Instruct** - Used via OpenRouter for synthetic data generation
3. **OpenAI** - Used for embeddings in vector search

### Key Dependencies
- `@langchain/langgraph` - For building stateful agent workflows
- `@langchain/anthropic` - Claude integration
- `@langchain/openai` - OpenAI integration
- `@langchain/mongodb` - MongoDB vector store integration
- `@langchain/langgraph-checkpoint-mongodb` - State persistence for conversations
- `mongodb` - MongoDB client
- `zod` - Runtime type validation
- `express` - Web server
- `dotenv` - Environment variable management

## Project Structure

```
langraph-agent/
├── index.ts           # Main Express server setup
├── agent.ts           # LangGraph agent implementation
├── seed-database.ts   # Database seeding script
├── .env              # Environment variables
└── package.json      # Project dependencies
```

### Component Details

- **index.ts**: Entry point that sets up the Express server and API endpoints for chat interactions
- **agent.ts**: Implements the LangGraph agent with tool definitions and conversation workflow
- **seed-database.ts**: Generates synthetic employee data using Mistral AI and seeds the MongoDB database

## Key Features

1. **Stateful Conversations**: Uses LangGraph and MongoDB to maintain conversation context
2. **Vector Search**: Implements semantic search over employee records using MongoDB Atlas Vector Search
3. **Rate Limiting**: Implements intelligent rate limiting with exponential backoff
4. **Tool-based Architecture**: Modular design with specialized tools for different operations
5. **Type Safety**: Comprehensive TypeScript types and Zod schemas for runtime validation

## Purpose and Use Cases

This system is designed to assist HR professionals and team leaders with:
1. Team building and composition
2. Employee information retrieval
3. Talent gap analysis
4. Resource allocation
5. Employee data management

The combination of multiple LLM models allows for:
- Cost-effective synthetic data generation (Mistral)
- High-quality reasoning and responses (Claude)
- Efficient semantic search capabilities (OpenAI embeddings)

## Environment Setup

The project requires several API keys and configuration values:
- `OPENAI_API_KEY` - For embeddings
- `OPENROUTER_API_KEY` - For accessing Mistral AI
- `ANTHROPIC_API_KEY` - For Claude access
- `MONGODB_ATLAS_URI` - MongoDB connection string
- `LANGSMITH_API_KEY` - For LangSmith tracing (optional)

## API Endpoints

- `GET /` - Health check endpoint
- `POST /chat` - Start a new conversation
- `POST /chat/:threadId` - Continue an existing conversation

## Getting Started

1. Clone the repository
2. Install dependencies: `npm install`
3. Set up environment variables in `.env`
4. Seed the database: `ts-node seed-database.ts`
5. Start the server: `ts-node index.ts`

The server will start on port 3000 by default. 