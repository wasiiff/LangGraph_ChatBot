const dotenv = require("dotenv");
dotenv.config();

const { ChatGoogleGenerativeAI } = require("@langchain/google-genai");
const { StateGraph, MessagesAnnotation } = require("@langchain/langgraph");
const readline = require("readline");
const chalk = require("chalk");

// --------------------- MODEL SETUP ---------------------
const model = new ChatGoogleGenerativeAI({
  apiKey: process.env.GEMINI_API_KEY,
  model: "models/gemini-2.0-flash", // ✅ Correct model name
  temperature: 0.7,
});

// --------------------- CLI STYLING ---------------------
const styles = {
  title: chalk.bold.cyan,
  user: chalk.bold.green,
  ai: chalk.bold.blue,
  system: chalk.dim.gray,
  error: chalk.bold.red,
  success: chalk.bold.green,
  warning: chalk.bold.yellow,
  info: chalk.bold.magenta,
  border: chalk.dim.white,
  sentiment: {
    positive: chalk.green,
    negative: chalk.red,
    neutral: chalk.yellow,
  },
};

const printBorder = () => console.log(styles.border("─".repeat(60)));
const printHeader = () => {
  console.clear();
  printBorder();
  console.log(styles.title("🚀 GEMINI LANGRAPH CHATBOT"));
  console.log(styles.system("Powered by Google Gemini 2.0 Flash + LangGraph"));
  printBorder();
  console.log(styles.info("Commands: 'exit' to quit, 'clear' to reset, 'help' for info"));
  printBorder();
  console.log();
};

// --------------------- HELPERS ---------------------
function formatMessages(messages) {
  return messages.map(m => `${m.role}: ${m.content}`).join("\n");
}

function extractText(response) {
  if (!response) return "";
  if (typeof response.content === "string") return response.content;
  if (Array.isArray(response.content) && response.content[0]?.text) {
    return response.content[0].text;
  }
  return String(response.content || "");
}

function displayState(state) {
  if (state.sentiment) {
    console.log(
      styles.system(
        `📊 Sentiment: ${styles.sentiment[state.sentiment](state.sentiment.toUpperCase())}`
      )
    );
  }
  if (state.calming) {
    console.log(styles.warning(`💚 Calming Response: ${state.calming}`));
  }
  if (state.summaries && state.summaries.length > 0) {
    console.log(styles.info(`📝 Summary added (${state.summaries.length} total)`));
  }
}

// --------------------- CHATBOT NODE ---------------------
const callModel = async (state) => {
  console.log(styles.system("🤖 Processing with Gemini..."));

  try {
    const formattedMessages = state.messages.map(msg => ({
      role: msg.role, // ✅ keep roles intact
      content: msg.content,
    }));

    const response = await model.invoke([
      { role: "system", content: "You are a helpful, friendly assistant. Be concise but informative." },
      ...formattedMessages,
    ]);

    const text = extractText(response);

    if (!text) {
      throw new Error("Empty response from model");
    }

    return {
      messages: [...state.messages, { role: "assistant", content: text }],
    };
  } catch (error) {
    console.error(styles.error("❌ Error calling model:"), error.message);
    return {
      messages: [
        ...state.messages,
        { role: "assistant", content: "Sorry, I encountered an error. Please try again." },
      ],
    };
  }
};

// --------------------- SUMMARIZER ---------------------
const summarizeNode = async (state) => {
  if (state.messages.length > 10) {
    console.log(styles.system("📝 Creating conversation summary..."));

    try {
      const response = await model.invoke([
        { role: "system", content: "Create a brief summary of the key points from this conversation:" },
        { role: "user", content: formatMessages(state.messages.slice(-8)) },
      ]);

      const text = extractText(response);

      return {
        summaries: [...(state.summaries || []), text],
        messages: state.messages,
      };
    } catch (error) {
      console.error(styles.error("❌ Summarization error:"), error.message);
    }
  }
  return state;
};

// --------------------- SENTIMENT ANALYSIS ---------------------
const sentimentNode = async (state) => {
  const lastUserMessage = state.messages.filter(m => m.role === "user").pop();
  if (!lastUserMessage) return state;

  try {
    const response = await model.invoke([
      { role: "system", content: "Analyze the sentiment of this message. Respond with only: positive, neutral, or negative" },
      { role: "user", content: lastUserMessage.content },
    ]);

    const sentiment = extractText(response).toLowerCase().trim();
    const validSentiments = ["positive", "neutral", "negative"];

    return {
      sentiment: validSentiments.includes(sentiment) ? sentiment : "neutral",
      messages: state.messages,
    };
  } catch (error) {
    console.error(styles.error("❌ Sentiment analysis error:"), error.message);
    return { sentiment: "neutral", messages: state.messages };
  }
};

// --------------------- CALMING RESPONSE ---------------------
const calmingNode = async (state) => {
  const lastUserMessage = state.messages.filter(m => m.role === "user").pop();
  if (!lastUserMessage || state.sentiment !== "negative") return state;

  try {
    const response = await model.invoke([
      { role: "system", content: "The user seems upset. Provide a brief, empathetic, and calming response." },
      { role: "user", content: lastUserMessage.content },
    ]);

    const text = extractText(response);

    return {
      calming: text,
      messages: state.messages,
    };
  } catch (error) {
    console.error(styles.error("❌ Calming response error:"), error.message);
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
      if (typeof result === "number" && isFinite(result)) {
        console.log(styles.success("🧮 Calculator activated!"));
        return {
          messages: [
            ...state.messages,
            { role: "assistant", content: `📊 **Calculation Result:** ${mathExpression} = **${result}**` },
          ],
        };
      }
    } catch (err) {
      console.log(styles.error("❌ Invalid mathematical expression"));
      return {
        messages: [
          ...state.messages,
          { role: "assistant", content: "Sorry, I couldn't calculate that expression. Please check your syntax." },
        ],
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
  if (/^[0-9+\-*/.() \s]+$/.test(content)) {
    return { ...state, route: "calculator" };
  }
  return { ...state, route: "chatbot" };
};

// --------------------- GRAPH DEFINITION ---------------------
const GraphState = MessagesAnnotation.with({
  sentiment: { default: () => "neutral" },
  calming: { default: () => null },
  summaries: { default: () => [] },
  route: { default: () => "chatbot" }
});

const workflow = new StateGraph(GraphState)
  .addNode("router", routerNode)
  .addNode("chatbot", callModel)
  .addNode("summarize", summarizeNode)
  .addNode("sentiment", sentimentNode)
  .addNode("calming", calmingNode)
  .addNode("calculator", calculatorNode);

workflow
  .addEdge("__start__", "router")
  .addConditionalEdges("router", (state) => {
    if (state.route === "calculator") return "calculator";
    return "chatbot";
  })
  .addEdge("calculator", "__end__")
  .addEdge("chatbot", "sentiment")
  .addConditionalEdges("sentiment", (state) => {
    if (state.sentiment === "negative") return "calming";
    return "summarize";
  })
  .addEdge("calming", "summarize")
  .addEdge("summarize", "__end__");

const app = workflow.compile();

// --------------------- ENHANCED CLI CHAT LOOP ---------------------
if (require.main === module) {
  if (!process.env.GEMINI_API_KEY) {
    console.error(styles.error("❌ GEMINI_API_KEY not found in environment variables!"));
    console.log(styles.system("Please create a .env file with your Gemini API key:"));
    console.log(styles.info("GEMINI_API_KEY=your_api_key_here"));
    process.exit(1);
  }

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    prompt: styles.user("You: "),
  });

  let conversationState = { 
    messages: [], 
    summaries: [], 
    sentiment: "neutral", 
    calming: null,
    route: "chatbot"
  };

  printHeader();

  const showHelp = () => {
    console.log(styles.info("\n📋 Available Commands:"));
    console.log(styles.system("  • exit/quit     - Exit the chatbot"));
    console.log(styles.system("  • clear/reset   - Clear conversation history"));
    console.log(styles.system("  • help          - Show this help message"));
    console.log(styles.system("  • stats         - Show conversation statistics"));
    console.log(styles.system("  • math expression - Use calculator (e.g., 2+2*3)"));
    console.log();
  };

  const showStats = () => {
    console.log(styles.info("\n📈 Conversation Statistics:"));
    console.log(styles.system(`  • Total messages: ${conversationState.messages.length}`));
    console.log(styles.system(`  • User messages: ${conversationState.messages.filter(m => m.role === "user").length}`));
    console.log(styles.system(`  • AI responses: ${conversationState.messages.filter(m => m.role === "assistant").length}`));
    console.log(styles.system(`  • Summaries created: ${conversationState.summaries?.length || 0}`));
    console.log();
  };

  const processInput = async (input) => {
    const command = input.toLowerCase().trim();

    switch (command) {
      case "exit":
      case "quit":
        console.log(styles.success("\n👋 Thanks for chatting! Goodbye!"));
        printBorder();
        rl.close();
        return;
      case "clear":
      case "reset":
        conversationState = { 
          messages: [], 
          summaries: [], 
          sentiment: "neutral", 
          calming: null,
          route: "chatbot"
        };
        printHeader();
        console.log(styles.success("🧹 Conversation cleared!\n"));
        rl.prompt();
        return;
      case "help":
        showHelp();
        rl.prompt();
        return;
      case "stats":
        showStats();
        rl.prompt();
        return;
      case "":
        rl.prompt();
        return;
    }

    try {
      const newState = {
        ...conversationState,
        messages: [...conversationState.messages, { role: "user", content: input }],
      };

      console.log(styles.system("🔄 Processing your message..."));
      const result = await app.invoke(newState);
      conversationState = result;

      const lastMessage = result.messages[result.messages.length - 1];
      if (lastMessage && lastMessage.role === "assistant") {
        console.log(styles.ai("\n🤖 AI: ") + lastMessage.content);
      }

      displayState(result);
      console.log();
    } catch (error) {
      console.error(styles.error("\n❌ An error occurred:"), error.message);
      console.error(styles.error("Stack trace:"), error.stack);
      console.log(styles.system("Please try again or type 'help' for assistance.\n"));
    }

    rl.prompt();
  };

  console.log(styles.system("💡 Tip: Try typing a math expression like '15*8+42' or ask me anything!"));
  console.log(styles.system("Type 'help' for more commands.\n"));

  rl.prompt();
  rl.on("line", (input) => processInput(input.trim()));
  rl.on("close", () => {
    console.log(styles.border("\n" + "─".repeat(60)));
    process.exit(0);
  });
}
