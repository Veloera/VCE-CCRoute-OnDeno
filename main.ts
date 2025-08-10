// deno-lint-ignore-file no-explicit-any
/**
 * @file claude_proxy.ts
 * @description A Deno script to serve an OpenAI-compatible API as an Anthropic Claude-compatible API.
 * This script is a single file with no external dependencies.
 *
 * It operates in two modes based on the ALLOWED_CLIENT_KEYS configuration:
 * 1. Fixed Key Mode: If ALLOWED_CLIENT_KEYS is not empty, it authenticates the client key
 *    against the list and uses the globally defined OPENAI_API_KEY for all upstream requests.
 * 2. Pass-through Key Mode: If ALLOWED_CLIENT_KEYS is empty, it forwards the client's
 *    provided API key directly to the upstream API.
 *
 * Key retrieval from client requests supports both 'Authorization: Bearer <key>' and 'x-api-key: <key>'.
 *
 * How to run:
 * deno run --allow-net --allow-env claude_proxy.ts
 */

import { STATUS_CODE } from "https://deno.land/std@0.224.0/http/status.ts";

// --- CONFIGURATION ---
// -----------------------------------------------------------------------------

const OPENAI_BASE_URL = Deno.env.get("OPENAI_BASE_URL") || "http://127.0.0.1:6240";
const OPENAI_API_KEY = Deno.env.get("OPENAI_API_KEY") || "YOUR_OPENAI_API_KEY_HERE";

const PORT = 6241;
const ALLOWED_CLIENT_KEYS: string[] = [];
const MODEL_WHITELIST: string[] = [];
const API_PREFIX = "";

// --- UTILITY FUNCTIONS ---
// -----------------------------------------------------------------------------

const createErrorResponse = (message: string, type: string, statusCode: number): Response => {
  return new Response(
    JSON.stringify({ type: "error", error: { type, message } }),
    { status: statusCode, headers: { "Content-Type": "application/json" } },
  );
};

const mapStopReason = (reason: string | null | undefined): string => {
  switch (reason) {
    case "stop": return "end_turn";
    case "length": return "max_tokens";
    case "tool_calls": return "tool_use";
    default: return "end_turn";
  }
};

/**
 * Extracts the API key from the request headers.
 * It checks for 'Authorization: Bearer <key>' first, then 'x-api-key'.
 * @param headers The request headers.
 * @returns The API key string or null if not found.
 */
const getClientApiKey = (headers: Headers): string | null => {
  const authHeader = headers.get("Authorization");
  if (authHeader && authHeader.startsWith("Bearer ")) {
    return authHeader.substring(7); // "Bearer ".length
  }
  return headers.get("x-api-key");
};


// --- API HANDLERS ---
// -----------------------------------------------------------------------------

async function handleModels(_req: Request, upstreamApiKey: string): Promise<Response> {
  const url = `${OPENAI_BASE_URL}/v1/models`;
  try {
    const response = await fetch(url, {
      headers: { "Authorization": `Bearer ${upstreamApiKey}` },
    });

    if (!response.ok) {
        const errorText = await response.text();
        console.error(`Upstream /models error: ${errorText}`);
        return createErrorResponse("Failed to fetch models from backend.", "api_error", response.status);
    }

    const openaiModels = await response.json();
    if (MODEL_WHITELIST.length > 0) {
      openaiModels.data = openaiModels.data.filter((model: any) =>
        MODEL_WHITELIST.includes(model.id)
      );
    }
    return new Response(JSON.stringify(openaiModels), { headers: { "Content-Type": "application/json" } });
  } catch (error) {
    console.error("Error in handleModels:", error);
    return createErrorResponse("Internal server error.", "api_error", 500);
  }
}

async function handleMessages(req: Request, upstreamApiKey: string): Promise<Response> {
  if (req.method !== "POST") {
    return createErrorResponse("Method not allowed.", "invalid_request_error", 405);
  }

  let claudeRequest;
  try {
    claudeRequest = await req.json();
  } catch {
    return createErrorResponse("Invalid JSON body.", "invalid_request_error", 400);
  }

  if (MODEL_WHITELIST.length > 0 && !MODEL_WHITELIST.includes(claudeRequest.model)) {
    return createErrorResponse(`Model '${claudeRequest.model}' is not available in the whitelist.`, "invalid_request_error", 400);
  }

  const openaiPayload: any = {
    model: claudeRequest.model,
    messages: [],
    max_tokens: claudeRequest.max_tokens,
    temperature: claudeRequest.temperature,
    top_p: claudeRequest.top_p,
    stop: claudeRequest.stop_sequences,
    stream: claudeRequest.stream || false,
  };

  if (claudeRequest.system) {
    openaiPayload.messages.push({
      role: "system",
      content: typeof claudeRequest.system === 'string' ? claudeRequest.system : claudeRequest.system.map((s: any) => s.text).join('\n')
    });
  }

  for (const message of claudeRequest.messages) {
    const textBlock = typeof message.content === 'string'
      ? { text: message.content }
      : message.content.find((block: any) => block.type === "text");
    if (textBlock) {
      openaiPayload.messages.push({ role: message.role, content: textBlock.text });
    }
  }

  const openaiResponse = await fetch(`${OPENAI_BASE_URL}/v1/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${upstreamApiKey}`,
    },
    body: JSON.stringify(openaiPayload),
  });

  if (!openaiResponse.ok) {
    const errorBody = await openaiResponse.text();
    console.error("Backend error:", errorBody);
    return createErrorResponse(`Backend API error: ${errorBody}`, "api_error", openaiResponse.status);
  }

  if (claudeRequest.stream) {
    const transformer = createOpenAiToClaudeStreamTransformer();
    const stream = openaiResponse.body!
      .pipeThrough(new TextDecoderStream())
      .pipeThrough(transformer)
      .pipeThrough(new TextEncoderStream());
    return new Response(stream, {
      headers: { "Content-Type": "text/event-stream", "Cache-Control": "no-cache", "Connection": "keep-alive" },
    });
  } else {
    const openaiJson = await openaiResponse.json();
    const claudeResponse = {
      id: openaiJson.id,
      type: "message",
      role: "assistant",
      model: openaiJson.model,
      content: [{ type: "text", text: openaiJson.choices[0].message.content }],
      stop_reason: mapStopReason(openaiJson.choices[0].finish_reason),
      stop_sequence: null,
      usage: {
        input_tokens: openaiJson.usage.prompt_tokens,
        output_tokens: openaiJson.usage.completion_tokens,
      },
    };
    return new Response(JSON.stringify(claudeResponse), { headers: { "Content-Type": "application/json" } });
  }
}

async function handleChatCompletions(req: Request, upstreamApiKey: string): Promise<Response> {
  if (req.method !== "POST") {
    return createErrorResponse("Method not allowed.", "invalid_request_error", 405);
  }

  let requestBody;
  try {
    requestBody = await req.json();
  } catch {
    return createErrorResponse("Invalid JSON body.", "invalid_request_error", 400);
  }

  if (MODEL_WHITELIST.length > 0 && !MODEL_WHITELIST.includes(requestBody.model)) {
    return createErrorResponse(`Model '${requestBody.model}' is not available in the whitelist.`, "invalid_request_error", 400);
  }

  const response = await fetch(`${OPENAI_BASE_URL}/v1/chat/completions`, {
    method: 'POST',
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${upstreamApiKey}`,
    },
    body: JSON.stringify(requestBody),
  });

  return new Response(response.body, { status: response.status, headers: response.headers });
}

function createOpenAiToClaudeStreamTransformer() {
  let isFirstChunk = true;
  let messageId = `msg_${crypto.randomUUID().replace(/-/g, "")}`;
  let model = "";
  let finalStopReason = "end_turn";
  let outputTokens = 0;

  return new TransformStream({
    transform(chunk, controller) {
      const lines = chunk.split("\n").filter(line => line.trim().startsWith("data:"));
      for (const line of lines) {
        const data = line.substring(5).trim();
        if (data === "[DONE]") continue;

        try {
          const json = JSON.parse(data);
          if (isFirstChunk) {
            isFirstChunk = false;
            model = json.model;
            const messageStart = {
              type: "message_start",
              message: {
                id: messageId, type: "message", role: "assistant", content: [], model: model,
                stop_reason: null, stop_sequence: null, usage: { input_tokens: 0, output_tokens: 1 },
              },
            };
            controller.enqueue(`event: message_start\ndata: ${JSON.stringify(messageStart)}\n\n`);
            const contentBlockStart = { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } };
            controller.enqueue(`event: content_block_start\ndata: ${JSON.stringify(contentBlockStart)}\n\n`);
          }

          const deltaText = json.choices[0]?.delta?.content;
          if (deltaText) {
            const contentBlockDelta = { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: deltaText } };
            controller.enqueue(`event: content_block_delta\ndata: ${JSON.stringify(contentBlockDelta)}\n\n`);
          }

          if (json.choices[0]?.finish_reason) {
            finalStopReason = mapStopReason(json.choices[0].finish_reason);
          }
          if (json.usage) {
            outputTokens = json.usage.completion_tokens;
          }
        } catch (error) {
          console.error("Error parsing stream chunk:", error, "Chunk:", data);
        }
      }
    },
    flush(controller) {
      const contentBlockStop = { type: "content_block_stop", index: 0 };
      controller.enqueue(`event: content_block_stop\ndata: ${JSON.stringify(contentBlockStop)}\n\n`);
      const messageDelta = {
        type: "message_delta",
        delta: { stop_reason: finalStopReason, stop_sequence: null },
        usage: { output_tokens: outputTokens || 1 },
      };
      controller.enqueue(`event: message_delta\ndata: ${JSON.stringify(messageDelta)}\n\n`);
      const messageStop = { type: "message_stop" };
      controller.enqueue(`event: message_stop\ndata: ${JSON.stringify(messageStop)}\n\n`);
    },
  });
}

// --- MAIN SERVER ---
// -----------------------------------------------------------------------------

async function mainHandler(req: Request): Promise<Response> {
  const url = new URL(req.url);
  let pathname = url.pathname;

  const prefix = API_PREFIX.startsWith("/") ? API_PREFIX : `/${API_PREFIX}`;
  if (API_PREFIX && pathname.startsWith(prefix)) {
    pathname = pathname.substring(prefix.length);
  } else if (API_PREFIX) {
    return createErrorResponse("Not Found", "not_found_error", 404);
  }

  // --- Key Handling Logic ---
  const clientKey = getClientApiKey(req.headers);
  let upstreamApiKey = "";

  if (ALLOWED_CLIENT_KEYS.length > 0) {
    // Mode 1: Fixed Key. Authenticate client and use server's key.
    if (!clientKey || !ALLOWED_CLIENT_KEYS.includes(clientKey)) {
      return createErrorResponse("Invalid or missing API key.", "authentication_error", 401);
    }
    upstreamApiKey = OPENAI_API_KEY;
  } else {
    // Mode 2: Pass-through Key. Use client's key for upstream.
    if (!clientKey) {
      return createErrorResponse("Client API key is required for upstream requests (from 'Authorization' or 'x-api-key' header).", "authentication_error", 401);
    }
    upstreamApiKey = clientKey;
  }
  
  // --- Routing ---
  switch (pathname) {
    case "/v1/models":
      return handleModels(req, upstreamApiKey);
    case "/v1/messages":
      return handleMessages(req, upstreamApiKey);
    case "/v1/chat/completions":
      return handleChatCompletions(req, upstreamApiKey);
    default:
      return createErrorResponse(`The requested endpoint '${pathname}' was not found.`, "not_found_error", STATUS_CODE.NotFound);
  }
}

// --- Server Startup ---
console.log(`🚀 Claude-compatible proxy server starting...`);
console.log(`Listening on http://localhost:${PORT}`);

if (ALLOWED_CLIENT_KEYS.length > 0) {
    console.log("🔒 Running in FIXED KEY mode.");
    console.log("   Clients must authenticate with a key from ALLOWED_CLIENT_KEYS.");
    console.log(`   Upstream requests will use the configured OPENAI_API_KEY.`);
    if (OPENAI_API_KEY === "YOUR_OPENAI_API_KEY_HERE") {
        console.warn("   ⚠️ WARNING: OPENAI_API_KEY is not set. Please configure it.");
    }
} else {
    console.log("🔄 Running in PASS-THROUGH KEY mode.");
    console.log("   The API key provided by the client (from 'Authorization' or 'x-api-key') will be forwarded.");
    console.log("   The configured OPENAI_API_KEY will be IGNORED.");
}

if (API_PREFIX) {
  console.log(`   API Prefix is set to: /${API_PREFIX}`);
}
if (MODEL_WHITELIST.length > 0) {
    console.log(`   Model whitelist enabled for: ${MODEL_WHITELIST.join(', ')}`);
}

Deno.serve({ port: PORT }, mainHandler);
