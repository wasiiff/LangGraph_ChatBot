const dotenv = require("dotenv");
dotenv.config();

const { ChatGoogleGenerativeAI } = require("@langchain/google-genai");
const { StateGraph, MessagesAnnotation } = require("@langchain/langgraph");
const readline = require("readline");

const model = new ChatGoogleGenerativeAI({
  apiKey: process.env.GEMINI_API_KEY,
  model: "models/gemini-2.0-flash",
});

const callModel = async (state) => {
  console.log("🔹 [callModel] Invoking Gemini with messages:", state.messages);

  const response = await model.invoke(state.messages);

  console.log("✅ [callModel] AI Response:", response.content);

  return { messages: [...state.messages, { role: "ai", content: response.content }] };
};

// 3. Summarization node (does not alter conversation flow)
const summarizeNode = async (state) => {
  console.log("🔹 [summarizeNode] Checking if summarization is needed...");

  if (state.messages.length > 5) {
    const summary = await model.invoke([
      { role: "system", content: "Summarize the following conversation briefly:" },
      { role: "user", content: JSON.stringify(state.messages) },
    ]);

    console.log("✅ [summarizeNode] Summary:", summary.content);

    return { summaries: [...(state.summaries || []), summary.content], messages: state.messages };
  }

  console.log("ℹ️ [summarizeNode] Not enough messages to summarize.");
  return state;
};


const sentimentNode = async (state) => {
  console.log("🔹 [sentimentNode] Analyzing sentiment...");

  const lastUserMessage = state.messages.filter(m => m.role === "user").pop();
  if (!lastUserMessage) return state;

  const analysis = await model.invoke([
    { role: "system", content: "Classify sentiment as positive, neutral, or negative." },
    { role: "user", content: lastUserMessage.content }
  ]);

  console.log("✅ [sentimentNode] Sentiment:", analysis.content);

  return { sentiment: analysis.content.toLowerCase(), messages: state.messages };
};


const calmingNode = async (state) => {
  console.log("🔹 [calmingNode] Providing calming response...");

  const lastUserMessage = state.messages.filter(m => m.role === "user").pop();
  if (!lastUserMessage) return state;

  const response = await model.invoke([
    { role: "system", content: "Respond empathetically and calm the user." },
    { role: "user", content: lastUserMessage.content }
  ]);

  console.log("✅ [calmingNode] Calming Response:", response.content);

  return { calming: response.content, messages: state.messages };
};

const graph = new StateGraph(MessagesAnnotation)
  .addNode("chatbot", callModel)
  .addNode("summarize", summarizeNode)
  .addNode("sentiment", sentimentNode)
  .addNode("calming", calmingNode)
  .addEdge("__start__", "chatbot")
  .addEdge("chatbot", "sentiment")
  .addEdge("sentiment", "calming", (state) => state.sentiment === "negative")
  .addEdge("sentiment", "summarize", (state) => state.messages.length > 5);

const app = graph.compile();

// 7. CLI setup
const rl = readline.createInterface({ input: process.stdin, output: process.stdout });

console.log("🤖 Advanced Gemini LangGraph Chatbot started! Type 'exit' to quit.");

async function ask(state = { messages: [] }) {
  rl.question("You: ", async (input) => {
    if (input.toLowerCase() === "exit") {
      rl.close();
      return;
    }

    console.log("📨 [User Input]:", input);

    const newState = await app.invoke({
      ...state,
      messages: [...state.messages, { role: "user", content: input }],
    });

    const lastMessage = newState.messages[newState.messages.length - 1];
    console.log("🤖 [AI]:", lastMessage.content);

    ask(newState);
  });
}

ask();
