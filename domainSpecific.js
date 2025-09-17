const dotenv = require("dotenv");
dotenv.config();

const { ChatGoogleGenerativeAI } = require("@langchain/google-genai");
const { StateGraph, MessagesAnnotation } = require("@langchain/langgraph");
const readline = require("readline");


const ALLOWED_DOMAIN = "sports";


const model = new ChatGoogleGenerativeAI({
  apiKey: process.env.GEMINI_API_KEY,
  model: "models/gemini-2.0-flash",
});




const domainCheckNode = async (state) => {
  const lastUserMessage = state.messages.filter(m => m.role === "user").pop();
  if (!lastUserMessage) return state;

  const check = await model.invoke([
    { role: "system", content: `Answer strictly with 'yes' or 'no'. Is the following query about ${ALLOWED_DOMAIN}?` },
    { role: "user", content: lastUserMessage.content }
  ]);

  const verdict = (check.content || "").toLowerCase().trim();
  console.log(`🔹 [domainCheckNode] Related to ${ALLOWED_DOMAIN}?`, verdict);

  if (!verdict.includes("yes")) {
    return {
      ...state,
      messages: [...state.messages, { role: "ai", content: `❌ I only respond to ${ALLOWED_DOMAIN}-related questions.` }],
      domainAllowed: false,
    };
  }

  return { ...state, domainAllowed: true };
};

const callModel = async (state) => {
  if (state.domainAllowed === false) return state;

  console.log("🔹 [callModel] Invoking Gemini with:", state.messages);
  const response = await model.invoke([
    { role: "system", content: `You are a helpful assistant that ONLY answers questions about ${ALLOWED_DOMAIN}. Stay strictly on topic.` },
    ...state.messages,
  ]);

  return { messages: [...state.messages, { role: "ai", content: response.content }] };
};

const summarizeNode = async (state) => {
  if (!state.domainAllowed) return state;

  if (state.messages.length > 5) {
    const summary = await model.invoke([
      { role: "system", content: `Summarize this ${ALLOWED_DOMAIN} conversation in 2–3 sentences:` },
      { role: "user", content: JSON.stringify(state.messages) },
    ]);

    console.log("✅ [summarizeNode] Conversation Summary:", summary.content);

    return { summaries: [...(state.summaries || []), summary.content], messages: state.messages };
  }

  return state;
};


const graph = new StateGraph(MessagesAnnotation)
  .addNode("domainCheck", domainCheckNode)
  .addNode("chatbot", callModel)
  .addNode("summarize", summarizeNode)
  .addEdge("__start__", "domainCheck")
  .addEdge("domainCheck", "chatbot")
  .addEdge("chatbot", "summarize");

const app = graph.compile();

const rl = readline.createInterface({ input: process.stdin, output: process.stdout });

console.log(`🤖 ${ALLOWED_DOMAIN.toUpperCase()}-Only Gemini Chatbot started! Type 'exit' to quit.`);

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
