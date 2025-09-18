const dotenv = require("dotenv");
dotenv.config();

const { ChatGoogleGenerativeAI } = require("@langchain/google-genai");
const { StateGraph, MessagesAnnotation } = require("@langchain/langgraph");
const readline = require("readline");
const chalk = require("chalk");

// --------------------- MODEL SETUP ---------------------
const model = new ChatGoogleGenerativeAI({
  apiKey: process.env.GEMINI_API_KEY,
  model: "models/gemini-2.0-flash",
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
  console.log(styles.title("🚀 ADVANCED GEMINI LANGGRAPH CHATBOT"));
  console.log(styles.system("Powered by Google Gemini 2.0 Flash + LangGraph"));
  printBorder();
  console.log(
    styles.info(
      "Commands: 'exit' to quit, 'clear' to reset, 'help' for info, 'graph' to view workflow"
    )
  );
  printBorder();
  console.log();
};

// --------------------- HELPERS ---------------------
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
        `📊 Sentiment: ${styles.sentiment[state.sentiment](
          state.sentiment.toUpperCase()
        )}`
      )
    );
  }
  if (state.calming) {
    console.log(styles.warning(`💚 Calming Response: ${state.calming}`));
  }
  if (state.summaries && state.summaries.length > 0) {
    console.log(
      styles.info(`📝 Summary added (${state.summaries.length} total)`)
    );
  }
}

// ------------------ Nodes ------------------

// Chatbot node
const callModel = async (state) => {
  console.log(styles.system("🤖 Processing with Gemini..."));
  try {
    const response = await model.invoke(state.messages);
    const text = extractText(response);
    return {
      messages: [...state.messages, { role: "assistant", content: text }],
    };
  } catch (err) {
    console.error(styles.error("❌ Model error:"), err.message);
    return {
      messages: [
        ...state.messages,
        { role: "assistant", content: "⚠️ Something went wrong." },
      ],
    };
  }
};

// Summarization node
const summarizeNode = async (state) => {
  if (state.messages.length > 5) {
    console.log(styles.system("📝 Creating summary..."));
    const summary = await model.invoke([
      {
        role: "system",
        content: "Summarize the following conversation briefly:",
      },
      { role: "user", content: JSON.stringify(state.messages) },
    ]);
    return {
      summaries: [...(state.summaries || []), extractText(summary)],
      messages: state.messages,
    };
  }
  return state;
};

// Sentiment node
const sentimentNode = async (state) => {
  const lastUserMessage = state.messages.filter((m) => m.role === "user").pop();
  if (!lastUserMessage) return state;

  const analysis = await model.invoke([
    {
      role: "system",
      content: "Classify sentiment as positive, neutral, or negative.",
    },
    { role: "user", content: lastUserMessage.content },
  ]);

  const sentiment = extractText(analysis).toLowerCase().trim();
  return { sentiment, messages: state.messages };
};

// Calming node
const calmingNode = async (state) => {
  const lastUserMessage = state.messages.filter((m) => m.role === "user").pop();
  if (!lastUserMessage) return state;

  const response = await model.invoke([
    { role: "system", content: "Respond empathetically and calm the user." },
    { role: "user", content: lastUserMessage.content },
  ]);

  return { calming: extractText(response), messages: state.messages };
};

const calculatorNode = async (state) => {
  const lastUserMessage = state.messages.filter((m) => m.role === "user").pop();
  if (!lastUserMessage) return state;

  let result;
  try {
    result = Function(`"use strict"; return (${lastUserMessage.content})`)();
    console.log(styles.success("🧮 Calculator activated!"));
  } catch (e) {
    result = "⚠️ Invalid math expression.";
  }

  return {
    messages: [
      ...state.messages,
      { role: "assistant", content: `📊 Result: ${result}` },
    ],
  };
};

// ------------------ Graph ------------------
const graph = new StateGraph(MessagesAnnotation)
  .addNode("router", async (state) => state) // router node just passes state
  .addNode("chatbot", callModel)
  .addNode("summarize", summarizeNode)
  .addNode("sentiment", sentimentNode)
  .addNode("calming", calmingNode)
  .addNode("calculator", calculatorNode)

  // Router → calculator if math
  .addEdge("router", "calculator", (state) => {
    const lastUserMessage = state.messages
      .filter((m) => m.role === "user")
      .pop();
    if (!lastUserMessage) return false;
    const expr = lastUserMessage.content.trim();
    return /^[0-9+\-*/().\s]+$/.test(expr);
  })

  // Router → chatbot if not math
  .addEdge("router", "chatbot", (state) => {
    const lastUserMessage = state.messages
      .filter((m) => m.role === "user")
      .pop();
    if (!lastUserMessage) return true;
    const expr = lastUserMessage.content.trim();
    return !/^[0-9+\-*/().\s]+$/.test(expr);
  })

  // Set __start__ → router (not directly to calculator/chatbot)
  .addEdge("__start__", "router")

  // Post-processing (chatbot only)
  .addEdge("chatbot", "sentiment")
  .addEdge("sentiment", "calming", (state) => state.sentiment === "negative")
  .addEdge("sentiment", "summarize", (state) => state.messages.length > 5);

const app = graph.compile();

// ------------------ CLI Setup ------------------
if (require.main === module) {
  if (!process.env.GEMINI_API_KEY) {
    console.error(styles.error("❌ GEMINI_API_KEY missing in .env file!"));
    process.exit(1);
  }

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    prompt: styles.user("You: "),
  });

  let conversationState = { messages: [] };

  printHeader();

  const showHelp = () => {
    console.log(styles.info("\n📋 Commands:"));
    console.log(styles.system("  • exit   - Quit chatbot"));
    console.log(styles.system("  • clear  - Reset conversation"));
    console.log(styles.system("  • help   - Show commands"));
    console.log(styles.system("  • graph  - View chatbot workflow diagram\n"));
  };

  const showGraph = () => {
    console.log(styles.title("\n🌐 LangGraph Workflow\n"));
    console.log(styles.border("─".repeat(60)));

    console.log(styles.ai("   ┌──────────┐"));
    console.log(styles.ai("   │ __start__│"));
    console.log(styles.ai("   └────┬─────┘"));
    console.log(styles.system("        │"));
    console.log(styles.system("        ├─▶ calculator (math only)"));
    console.log(styles.system("        │"));
    console.log(styles.system("        ▼"));
    console.log(styles.ai("     chatbot"));
    console.log(styles.system("        │"));
    console.log(styles.system("        ▼"));
    console.log(styles.ai("     sentiment"));
    console.log(styles.system("     ├──▶ calming (if NEGATIVE)"));
    console.log(styles.system("     └──▶ summarize (if >5 msgs)"));
    console.log(styles.border("─".repeat(60)));
    console.log();
  };

  const processInput = async (input) => {
    const command = input.toLowerCase().trim();

    switch (command) {
      case "exit":
        console.log(styles.success("\n👋 Goodbye!"));
        rl.close();
        return;
      case "clear":
        conversationState = { messages: [] };
        printHeader();
        console.log(styles.success("🧹 Conversation cleared!\n"));
        rl.prompt();
        return;
      case "help":
        showHelp();
        rl.prompt();
        return;
      case "graph":
        showGraph();
        rl.prompt();
        return;
    }

    const newState = await app.invoke({
      ...conversationState,
      messages: [
        ...conversationState.messages,
        { role: "user", content: input },
      ],
    });

    conversationState = newState;
    const lastMessage = newState.messages[newState.messages.length - 1];
    if (lastMessage) {
      console.log(styles.ai("\n🤖 AI: ") + lastMessage.content);
    }
    displayState(newState);
    console.log();
    rl.prompt();
  };

  console.log(
    styles.system("💡 Tip: Try a math expression like '12*4+6' or just chat!")
  );
  console.log(
    styles.system("💡 Type 'graph' anytime to view chatbot workflow.\n")
  );

  rl.prompt();
  rl.on("line", (input) => processInput(input.trim()));
  rl.on("close", () => {
    printBorder();
    process.exit(0);
  });
}
