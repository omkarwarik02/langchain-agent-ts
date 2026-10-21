import "dotenv/config";
import { createAgent,modelRetryMiddleware,
  toolCallLimitMiddleware,
  contextEditingMiddleware,
  ClearToolUsesEdit, } from "langchain";
import { ChatGroq } from "@langchain/groq";
import { AIMessage, BaseMessage, HumanMessage, ToolMessage } from "@langchain/core/messages";
import { researchTools } from "./tools.js";
import { resolve } from "node:dns";

const model = new ChatGroq({
    model:"openai/gpt-oss-120b",
    temperature:0.2,
});

const agent = createAgent({
    model,
    tools: researchTools,
    middleware:[ 
        modelRetryMiddleware({
            maxRetries:3,
            initialDelayMs:5000,
            backoffFactor:2,
        }),
        toolCallLimitMiddleware({ runLimit:6}),
        toolCallLimitMiddleware({ toolName:"fetch_page", runLimit:2}),

        contextEditingMiddleware({
            edits:[new ClearToolUsesEdit({ trigger:{tokens:3000}, keep:{ messages: 2}})],
        }),
    
    ],
    systemPrompt:
        `Today's date is ${new Date().toDateString()}. ` +
    "You are a research assistant. For anything about current versions, news or facts that change over time, use web_search. " +
    "If the search snippets are not detailed enough, use fetch_page on the most relevant URL to read the full page. " +
    "Never answer those from memory. If a tool fails, say so clearly instead of guessing. Cite sources as plain URLs. " +
    "Be efficient: use as few tool calls as needed. Once you have read an official source that answers the question, stop searching and answer.",
});

    const sleep =(ms:number) => new Promise((resolve)=> setTimeout(resolve, ms));

    async function invokeWithRetry(messages:BaseMessage[], maxRetries = 2){
        for(let attempt = 0; ; attempt++){
            try{
                return await model.invoke(messages);
            } catch(err){
                 const  isRateLimit = err instanceof Error && err.message.includes("429");
                 if(!isRateLimit || attempt >= maxRetries) throw err;

                 console.log(`⏳ Rate limited. Waiting 20s, then retry ${attempt + 1}/${maxRetries}...`);
                 await sleep(20_000);
            }
        }
    }

















const question = "What's the latest version of React and what changed?";
 let seen = 0; 
    let answered = false;
    let lastMessages: BaseMessage[] = [];

try{
    const stream = await agent.stream(
        {messages:[{ role:"user", content: question}]},
        {streamMode:"values", recursionLimit: 30},
    )
   


    for await (const state of stream) {
        lastMessages = state.messages
        const newMessages = state.messages.slice(seen);
        seen = state.messages.length;

        for(const msg of newMessages){
            if(AIMessage.isInstance(msg) && msg.tool_calls?.length){
                for(const call of msg.tool_calls){
                     console.log(`🔧 ${call.name}(${JSON.stringify(call.args)})`);
                }
            } else if(ToolMessage.isInstance(msg)){
                 console.log(`📄 ${msg.name} → ${String(msg.content).slice(0, 120)}`);
            } else if(AIMessage.isInstance(msg)){
                console.log(`\n🤖 ${msg.content}`);
                answered = true
            }
        }
        
    }
    if(!answered){
        console.log("\n⚠️ The agent stopped before writing a final answer (tool limit reached).");

        try {
        const final = await invokeWithRetry([
            ...lastMessages,
            new HumanMessage(
                 "You have no tool calls left. Answer the original question now, using only the information you have already gathered. Cite sources as plain URLs.",
            ),
        ]);
         console.log(`\n🤖 ${final.content}`);
    } catch(err){
        console.log(`\n⛔ Could not produce a final answer: ${err instanceof Error ? err.message : String(err)}`);
  }
    }


} catch(err){
     console.log(`⛔ Agent stopped: ${err instanceof Error ? err.message : String(err)}`);
}