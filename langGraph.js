const dotenv = require("dotenv");
dotenv.config();

const { ChatGoogleGenerativeAI } = require("@langchain/google-genai");
const { StateGraph, MessagesAnnotation } = require("@langchain/langgraph");

// --------------------- MODEL SETUP ---------------------
const model = new ChatGoogleGenerativeAI({
  apiKey: process.env.GEMINI_API_KEY,
  model: "models/gemini-2.0-flash",
});

// Helper: format messages for model input
function formatMessages(messages) {
  return messages.map(m => `${m.role}: ${m.content}`).join("\n");
}

// --------------------- CHATBOT NODE ---------------------
const callModel = async (state) => {
  console.log("🤖 Processing with Gemini...");

  try {
    const response = await model.invoke([
      { role: "system", content: "You are a helpful, friendly assistant. Be concise but informative." },
      { role: "user", content: formatMessages(state.messages) },
    ]);

    return { 
      messages: [...state.messages, { role: "assistant", content: response.content }] 
    };
  } catch (error) {
    console.error("❌ Error calling model:", error.message);
    return { 
      messages: [...state.messages, { 
        role: "assistant", 
        content: "Sorry, I encountered an error. Please try again." 
      }] 
    };
  }
};

// --------------------- SUMMARIZER ---------------------
const summarizeNode = async (state) => {
  if (state.messages.length > 10) {
    console.log("📝 Creating conversation summary...");
    
    try {
      const summary = await model.invoke([
        { 
          role: "system", 
          content: "Create a brief summary of the key points from this conversation:" 
        },
        { role: "user", content: formatMessages(state.messages.slice(-8)) }
      ]);

      return { 
        summaries: [...(state.summaries || []), summary.content], 
        messages: state.messages 
      };
    } catch (error) {
      console.error("❌ Summarization error:", error.message);
    }
  }
  return state;
};

// --------------------- SENTIMENT ANALYSIS ---------------------
const sentimentNode = async (state) => {
  const lastUserMessage = state.messages.filter(m => m.role === "user").pop();
  if (!lastUserMessage) return state;

  try {
    const analysis = await model.invoke([
      { 
        role: "system", 
        content: "Analyze the sentiment of this message. Respond with only: positive, neutral, or negative" 
      },
      { role: "user", content: lastUserMessage.content }
    ]);

    const sentiment = analysis.content.toLowerCase().trim();
    const validSentiments = ['positive', 'neutral', 'negative'];
    
    return { 
      sentiment: validSentiments.includes(sentiment) ? sentiment : 'neutral',
      messages: state.messages 
    };
  } catch (error) {
    console.error("❌ Sentiment analysis error:", error.message);
    return { sentiment: 'neutral', messages: state.messages };
  }
};

// --------------------- CALMING RESPONSE ---------------------
const calmingNode = async (state) => {
  const lastUserMessage = state.messages.filter(m => m.role === "user").pop();
  if (!lastUserMessage || state.sentiment !== 'negative') return state;

  try {
    const response = await model.invoke([
      { 
        role: "system", 
        content: "The user seems upset. Provide a brief, empathetic, and calming response." 
      },
      { role: "user", content: lastUserMessage.content }
    ]);

    return { 
      calming: response.content,
      messages: state.messages 
    };
  } catch (error) {
    console.error("❌ Calming response error:", error.message);
    return state;
  }
};

// --------------------- CALCULATOR TOOL ---------------------
const calculatorNode = async (state) => {
  const lastUserMessage = state.messages.filter(m => m.role === "user").pop();
  if (!lastUserMessage) return state;

  const mathExpression = lastUserMessage.content.trim();
  const validMathRegex = /^[0-9+\-*/.() \s]+$/;
  
  if (validMathRegex.test(mathExpression)) {
    try {
      const result = Function(`"use strict"; return (${mathExpression})`)();
      
      if (typeof result === 'number' && isFinite(result)) {
        console.log("🧮 Calculator activated!");
        return { 
          messages: [...state.messages, { 
            role: "assistant", 
            content: `📊 **Calculation Result:** ${mathExpression} = **${result}**` 
          }] 
        };
      }
    } catch (err) {
      console.log("❌ Invalid mathematical expression");
      return { 
        messages: [...state.messages, { 
          role: "assistant", 
          content: "Sorry, I couldn't calculate that expression. Please check your syntax." 
        }] 
      };
    }
  }

  return state;
};

// --------------------- ROUTER LOGIC ---------------------
const routerNode = async (state) => {
  const lastUserMessage = state.messages.filter(m => m.role === "user").pop();
  if (!lastUserMessage) return state;

  const content = lastUserMessage.content.trim();
  
  // Check if it's a math expression
  if (/^[0-9+\-*/.() \s]+$/.test(content)) {
    return { ...state, route: "calculator" };
  }
  
  return { ...state, route: "chatbot" };
};

// --------------------- GRAPH DEFINITION ---------------------
const workflow = new StateGraph(MessagesAnnotation)
  .addNode("router", routerNode)
  .addNode("chatbot", callModel)
  .addNode("summarize", summarizeNode)
  .addNode("sentiment", sentimentNode)
  .addNode("calming", calmingNode)
  .addNode("calculator", calculatorNode);

// Add edges
workflow
  .addEdge("__start__", "router")
  .addConditionalEdges("router", {
    calculator: (state) => state.route === "calculator",
    chatbot: (state) => state.route === "chatbot",
  })
  .addEdge("calculator", "__end__")
  .addEdge("chatbot", "sentiment")
  .addConditionalEdges("sentiment", {
    calming: (state) => state.sentiment === "negative",
    summarize: (state) => state.sentiment !== "negative",
  })
  .addEdge("calming", "summarize")
  .addEdge("summarize", "__end__");

const app = workflow.compile();

// Export for LangGraph Studio
module.exports = app;