import "dotenv/config";
import {
  StateSchema,
  MessagesValue,
  ReducedValue,
  StateGraph,
  START,
  END,
} from "@langchain/langgraph";
import type { GraphNode } from "@langchain/langgraph";
import { ChatGroq } from "@langchain/groq";
import {
  HumanMessage,
  SystemMessage,
  ToolMessage,
  AIMessage,
} from "@langchain/core/messages";
import { z } from "zod";
import { researchTools } from "./tools.js";
import type { StructuredToolInterface } from "@langchain/core/tools";




const AgentState = new StateSchema({
  messages: MessagesValue,

  toolCallCount: new ReducedValue(z.number().default(0), {
    reducer: (current, added) => current + added,
  }),
});

const SYSTEM_PROMPT =
  `Today's date is ${new Date().toDateString()}. ` +
  "You are a research assistant. For anything about current versions, news or facts that change over time, use web_search. " +
  "If the search snippets are not detailed enough, use fetch_page on the most relevant URL to read the full page. " +
  "Never answer those from memory. If a tool fails, say so clearly instead of guessing. Cite sources as plain URLs. " +
  "Be efficient: use as few tool calls as needed. Once you have read an official source that answers the question, stop searching and answer.";

const model = new ChatGroq({
  model: "openai/gpt-oss-120b",
  temperature: 0,
});

const modeWithTools = model.bindTools(researchTools);

const modelRetryPolicy = {
  maxAttempts: 3,
  initialInterval:10_000,
  backoffFactor:2,
  retryOn:(err:unknown) => err instanceof Error && err.message.includes("429")
}

const agentNode: GraphNode<typeof AgentState> = async (state) => {
  const response = await modeWithTools.invoke([
    new SystemMessage(SYSTEM_PROMPT),
    ...state.messages,
  ]);
  return { messages: [response] };
};

const finalAnswerNode: GraphNode<typeof AgentState> = async (state) => {
  const response = await model.invoke([
    new SystemMessage(SYSTEM_PROMPT),
    ...state.messages,
    new HumanMessage(
      "You have no tool calls left. Answer the original question now, using only the information you have already gathered. Cite sources as plain URLs.",
    ),
  ]);
  return { messages: [response] };
};

const toolsByName: Record<string, StructuredToolInterface> = Object.fromEntries(
  researchTools.map((t) => [t.name, t]),
);

const callKey = (call: { name: string; args: unknown }) =>
  `${call.name}:${JSON.stringify(call.args)}`;

const toolsNode: GraphNode<typeof AgentState> = async (state) => {
  const last = state.messages.at(-1);
  if (!last || !AIMessage.isInstance(last) || !last.tool_calls?.length) {
    return {};
  }

  const seen = new Set<string>();
  for (const msg of state.messages.slice(0, -1)) {
    if (AIMessage.isInstance(msg)) {
      for (const call of msg.tool_calls ?? []) seen.add(callKey(call));
    }
  }

  const results: ToolMessage[] = [];
  for (const call of last.tool_calls) {
    const key = callKey(call);
    let content: string;

    if (seen.has(key)) {
      content =
        "You already made this exact call. Use the earlier result instead of repeating it.";
    } else {
      seen.add(key);
      const tool = toolsByName[call.name];

      if (!tool) {
        content = `Error: unknown tool "${call.name}". Available tools: ${Object.keys(toolsByName).join(", ")}.`;
      } else {
        try {
          content = String(await tool.invoke(call.args));
        } catch (err) {
          content = `Error: ${call.name} failed: ${err instanceof Error ? err.message : String(err)}`;
        }
      }
    }

    results.push(
      new ToolMessage({
        content,
        tool_call_id: call.id ?? "",
        name: call.name,
      }),
    );
  }
  return { messages: results, toolCallCount: last.tool_calls.length };
};

const MAX_TOOL_CALLS = 6;

const routeAfterAgent = (state: typeof AgentState.State) => {
  const last = state.messages.at(-1);
  if (last && AIMessage.isInstance(last) && last.tool_calls?.length) {
    return "tools";
  }
  return END;
};

const routeAfterTools = (state: typeof AgentState.State) => {
  if (state.toolCallCount >= MAX_TOOL_CALLS) {
    return "final_answer";
  }
  return "agent";
};

export const graph = new StateGraph(AgentState)
  .addNode("agent", agentNode, { retryPolicy:modelRetryPolicy})
  .addNode("tools", toolsNode)
  .addNode("final_answer", finalAnswerNode,{ retryPolicy:modelRetryPolicy})
  .addEdge(START, "agent")
  .addConditionalEdges("agent", routeAfterAgent, ["tools", END])
  .addConditionalEdges("tools", routeAfterTools, ["agent", "final_answer"])
  .addEdge("final_answer", END)
  .compile();
