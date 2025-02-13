import { OpenAIEmbeddings } from "@langchain/openai";
import { StructuredOutputParser } from "@langchain/core/output_parsers";
import { MongoClient } from "mongodb";
import { MongoDBAtlasVectorSearch } from "@langchain/mongodb";
import { z } from "zod";
import "dotenv/config";
import { ChatOpenAI } from "@langchain/openai";

const client = new MongoClient(process.env.MONGODB_ATLAS_URI as string);

// Configure ChatOpenAI to use OpenRouter with a more affordable model
const llm = new ChatOpenAI({
  configuration: {
    baseURL: "https://openrouter.ai/api/v1",
    apiKey: process.env.OPENROUTER_API_KEY,
    defaultHeaders: {
      "HTTP-Referer": "http://localhost:3000",
      "X-Title": "HR Database App",
    },
  },
  modelName: "mistralai/mistral-7b-instruct", // Using Mistral, a reliable and cost-effective model
  temperature: 0.5,
  maxTokens: 1000,
  maxRetries: 3,
});

const EmployeeSchema = z.object({
  employee_id: z.string(),
  first_name: z.string(),
  last_name: z.string(),
  job_title: z.string(),
  department: z.string(),
  email: z.string().email(),
  phone_number: z.string(),
  hire_date: z.string(),
  is_remote: z.boolean(),
  notes: z.string(),
});

type Employee = z.infer<typeof EmployeeSchema>;

const parser = StructuredOutputParser.fromZodSchema(z.array(EmployeeSchema));

// Enhanced rate limiter with exponential backoff
class RateLimiter {
  private queue: (() => Promise<any>)[] = [];
  private processing = false;
  private lastCallTime = 0;
  private retryCount = 0;
  private maxRetries = 3;
  
  constructor(private rateLimit: number) {}

  async execute<T>(fn: () => Promise<T>): Promise<T> {
    return new Promise((resolve, reject) => {
      this.queue.push(async () => {
        try {
          const now = Date.now();
          const timeSinceLastCall = now - this.lastCallTime;
          if (timeSinceLastCall < this.rateLimit) {
            const waitTime = this.rateLimit - timeSinceLastCall;
            await new Promise(resolve => setTimeout(resolve, waitTime));
          }
          
          this.lastCallTime = Date.now();
          const result = await this.retryWithBackoff(fn);
          this.retryCount = 0; // Reset retry count on success
          resolve(result);
        } catch (error) {
          reject(error);
        }
      });
      
      this.processQueue();
    });
  }

  private async retryWithBackoff(fn: () => Promise<any>): Promise<any> {
    try {
      return await fn();
    } catch (error: any) {
      if (error.message?.includes('429') && this.retryCount < this.maxRetries) {
        this.retryCount++;
        const backoffTime = Math.pow(2, this.retryCount) * 1000; // Exponential backoff
        console.log(`Rate limited. Retrying in ${backoffTime/1000} seconds...`);
        await new Promise(resolve => setTimeout(resolve, backoffTime));
        return this.retryWithBackoff(fn);
      }
      throw error;
    }
  }

  private async processQueue() {
    if (this.processing || this.queue.length === 0) return;
    
    this.processing = true;
    while (this.queue.length > 0) {
      const task = this.queue.shift();
      if (task) await task();
    }
    this.processing = false;
  }
}

const rateLimiter = new RateLimiter(2000); // Increased rate limit to 2 seconds

async function generateSyntheticData(): Promise<Employee[]> {
  const prompt = `Generate 3 fictional employee records with realistic details. Return a structured JSON format. Make sure to follow the exact format specified.

  ${parser.getFormatInstructions()}`;

  console.log("Generating synthetic data...");

  try {
    const response = await llm.invoke(prompt);
    return parser.parse(response.content as string);
  } catch (error: any) {
    if (error.message?.includes('429')) {
      console.error("Rate limit exceeded. Please wait a moment and try again.");
    } else {
      console.error("Error generating data:", error.message);
    }
    return [];
  }
}

async function createEmployeeSummary(employee: Employee): Promise<string> {
  return `${employee.first_name} ${employee.last_name} works as a ${employee.job_title} in ${employee.department}. 
  Email: ${employee.email}, Phone: ${employee.phone_number}. 
  Hired on: ${employee.hire_date}, Remote: ${employee.is_remote ? "Yes" : "No"}.
  Notes: ${employee.notes}`;
}

async function seedDatabase(): Promise<void> {
  try {
    await client.connect();
    console.log("Connected to MongoDB!");

    const db = client.db("hr_database");
    const collection = db.collection("employees");

    await collection.deleteMany({});

    const syntheticData = await generateSyntheticData();

    if (syntheticData.length === 0) {
      console.log("No data generated. Retrying in 5 seconds...");
      await new Promise(resolve => setTimeout(resolve, 5000));
      return seedDatabase();
    }

    for (const record of syntheticData) {
      await rateLimiter.execute(async () => {
        const summary = await createEmployeeSummary(record);
        
        // For vector search, use a simpler approach without embeddings
        await collection.insertOne({
          ...record,
          summary,
          created_at: new Date()
        });

        console.log("Successfully saved:", record.employee_id);
      });
    }

    console.log("Database seeding completed.");

  } catch (error) {
    console.error("Error seeding database:", error);
    if (error instanceof Error && error.message.includes('429')) {
      console.log("Rate limit hit. Waiting 5 seconds before retry...");
      await new Promise(resolve => setTimeout(resolve, 5000));
      return seedDatabase();
    }
  } finally {
    await client.close();
  }
}

seedDatabase().catch(console.error);


// Used OpenRouter (via Mistral AI model) to generate realistic employee records
// Used Zod for runtime type validation
// Implements queue-based rate limiting to prevent API overload
