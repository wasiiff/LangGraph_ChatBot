const { StateGraph, MessagesAnnotation } = require("@langchain/langgraph");

const helloNode = async (state) => {
  return {
    messages: [
      ...(state.messages ?? []),
      {
        role: "ai",
        content: `   
Hello From LangGraph!
This is a LangGraph Hello World Tree
`,
      },
    ],
  };
};

const graph = new StateGraph(MessagesAnnotation)
  .addNode("hello", helloNode)
  .addEdge("__start__", "hello");

const app = graph.compile();

(async function main() {
  try {
    const result = await app.invoke({ messages: [] });
    console.log(result.messages[result.messages.length - 1].content);
  } catch (error) {
    console.log("Error Occurred:", error);
    process.exit(1);
  }
})();
