import "dotenv/config";
import { tool, createAgent } from "langchain";
import { z } from "zod";
import { ChatGroq } from "@langchain/groq";

import { getTime} from "./tools.js"


const model = new ChatGroq({
    model:"openai/gpt-oss-120b",
    temperature:0.2,
});

const agent = createAgent({
    model,
    tools:[getTime],
    systemPrompt:"You are a helpful assistant. Use tools when you need real-time information."
})

const result = await agent.invoke({
    messages:[{ role:"user", content:"What time is it in Mumbai and in New York?"}],
});

for (const msg of result.messages){
    console.log(`[${msg.type}]`, msg.content || "(tool call)");
}
