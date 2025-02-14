import 'dotenv/config';
import express, { Express, Request, Response } from "express";
import { MongoClient } from "mongodb";
import { callAgent } from './agent';

const app: Express = express();
app.use(express.json());

// Initialize MongoDB client
const client = new MongoClient(process.env.MONGODB_ATLAS_URI as string);

async function startServer() {
  try {
    await client.connect();
    await client.db("admin").command({ ping: 1 });
    console.log("Pinged your deployment. You successfully connected to MongoDB!");

    // Set up basic Express route
    app.get('/', (req: Request, res: Response) => {
      res.send('LangGraph Agent Server');
    });

    // API endpoint to start a new conversation
    app.post('/chat', async (req: Request, res: Response) => {
      const initialMessage = req.body.message;
      const threadId = Date.now().toString();
      try {
        const response = await callAgent(client, initialMessage, threadId);
        res.json({ threadId, response });
      } catch (error: any) {
        console.error('Error starting conversation:', error);
        // Enhanced error handling for OpenRouter-specific errors
        if (error.message?.includes('429')) {
          res.status(429).json({ error: 'Rate limit exceeded. Please try again later.' });
        } else if (error.message?.includes('401')) {
          res.status(401).json({ error: 'Authentication failed. Please check your API key.' });
        } else {
          res.status(500).json({ error: 'Internal server error' });
        }
      }
    });

    // API endpoint to send a message in an existing conversation
    app.post('/chat/:threadId', async (req: Request, res: Response) => {
      const { threadId } = req.params;
      const { message } = req.body;
      try {
        const response = await callAgent(client, message, threadId);
        res.json({ response });
      } catch (error: any) {
        console.error('Error in chat:', error);
        // Enhanced error handling for OpenRouter-specific errors
        if (error.message?.includes('429')) {
          res.status(429).json({ error: 'Rate limit exceeded. Please try again later.' });
        } else if (error.message?.includes('401')) {
          res.status(401).json({ error: 'Authentication failed. Please check your API key.' });
        } else {
          res.status(500).json({ error: 'Internal server error' });
        }
      }
    });

    const PORT = process.env.PORT || 3000;
    app.listen(PORT, () => {
      console.log(`Server running on port ${PORT}`);
    });
  } catch (error) {
    console.error('Error connecting to MongoDB:', error);
    process.exit(1);
  }
}

startServer();