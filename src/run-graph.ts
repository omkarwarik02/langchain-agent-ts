import { graph } from "../src/graph-agent.js";
import { AIMessage, ToolMessage, type BaseMessage } from "@langchain/core/messages";

const question = "What's the latest version of React and what changed?";

try {
  const stream = await graph.stream(
    { messages: [{ role: "user", content: question }] },
    { streamMode: "updates", recursionLimit: 20 },
  );

  for await (const chunk of stream) {
    for (const [nodeName, update] of Object.entries(chunk)) {
      console.log(`\n── ${nodeName} ──`);

      const messages = (update as { messages?: BaseMessage[] }).messages ?? [];

      for (const msg of messages) {
        if (AIMessage.isInstance(msg) && msg.tool_calls?.length) {
          for (const call of msg.tool_calls) {
            console.log(`🔧 ${call.name}(${JSON.stringify(call.args)})`);
          }
        } else if (ToolMessage.isInstance(msg)) {
          console.log(`📄 ${msg.name} → ${String(msg.content).slice(0, 120)}`);
        } else if (AIMessage.isInstance(msg)) {
          console.log(`🤖 ${msg.content}`);
        }
      }
    }
  }
} catch (err) {
  console.log(`⛔ Graph stopped: ${err instanceof Error ? err.message.slice(0, 200) : String(err)}`);
}