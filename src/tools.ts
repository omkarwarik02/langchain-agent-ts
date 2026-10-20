import { tool } from "langchain";
import { z } from "zod";


 export const getTime = tool(
    async ({ timezone}) => new Date().toLocaleString("en-IN",{ timeZone: timezone}),
    {
        name:"get_time",
        description:"Get the current date and time in a given timezone.",
        schema:z.object({
            timezone:z.string().describe("IANA timezone name, for example 'Asia/Kolkata'"),
        }),
    },
);





export const webSearch = tool(
    async({ query}) => {
        try{
            const res = await fetch("https://api.tavily.com/search",{
                method:"POST",
                headers:{
                    "Content-Type": "application/json",
                    Authorization:`Bearer ${process.env.TAVILY_API_KEY}`,
                },
                body:JSON.stringify({ query, max_results:3}),
            });
            if(!res.ok) return `Error: Tavily returned ${res.status}`;

            const data = (await res.json()) as {
                results:{title:string; url:string; content:string} [];
            };
            return data.results
                .map((r,i)=> `${i + 1}. ${r.title}\n${r.url}\n${r.content.slice(0, 500)}`).join("\n\n");

        } catch(err){
            return `Error: web_search failed: ${err instanceof Error ? err.message: String(err)}`;
        }
    },
    {
        name:"web_search",
        description:"Search the web for up-to-date information. Returns title, URLS and short snippets.",
        schema: z.object({
            query: z.string().describe("A short web search query, like 'latest React version release notes'")
        }),
    },
);

const MAX_PAGE_CHARS = 3000;

export const fetchPage = tool(
    async ({ url }) => {
        try{
            const res = await fetch(url,{
                headers:{"User-Agent":"Mozilla/5.0(langchain-agent.ts)"},
                signal:AbortSignal.timeout(10_000),
            });
            if(!res.ok) return `Error: HTTP ${res.status} for ${url}`

            const text = (await res.text())
                 .replace(/<script[\s\S]*?<\/script>/gi, " ")
        .replace(/<style[\s\S]*?<\/style>/gi, " ")
        .replace(/<[^>]+>/g, " ")
        .replace(/\s+/g, " ")
        .trim();

        return text.length > MAX_PAGE_CHARS ? text.slice(0, MAX_PAGE_CHARS) + "...[page truncated: this is all of the page you can read. Fetching it again returns the same text. Use what is here, or try a different URL.]" : text;

        }catch(err)  {
             return `Error: fetch_page failed: ${err instanceof Error ? err.message : String(err)}`;
    }
        },
        {
            name: "fetch_page",
             description: "Read the main text of a web page. Use it on a URL from web_search when the snippet is not detailed enough.",
             schema:z.object({
                url:z.url().describe("Full URL of the page to read, usually taken from web_search results"),
             })
        }
    
)
export const researchTools = [getTime, webSearch, fetchPage]