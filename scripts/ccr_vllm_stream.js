"use strict";

function sanitizeDataLine(line, toolChoices) {
  if (!line.startsWith("data:")) {
    return line;
  }
  const prefixEnd = line.indexOf(":") + 1;
  const payload = line.slice(prefixEnd).trimStart();
  if (!payload || payload === "[DONE]") {
    return line;
  }

  let event;
  try {
    event = JSON.parse(payload);
  } catch {
    return line;
  }
  if (!Array.isArray(event.choices)) {
    return line;
  }

  event.choices = event.choices.filter((choice) => {
    const choiceIndex = Number(choice.index || 0);
    const delta = choice.delta;
    if (!delta || typeof delta !== "object") {
      return true;
    }
    if (Array.isArray(delta.tool_calls) && delta.tool_calls.length) {
      toolChoices.add(choiceIndex);
    }
    if (toolChoices.has(choiceIndex) && typeof delta.content === "string") {
      delete delta.content;
    }
    const hasDelta = Object.keys(delta).length > 0;
    return hasDelta || choice.finish_reason != null;
  });

  if (!event.choices.length && event.usage == null) {
    return "";
  }
  return `data: ${JSON.stringify(event)}`;
}

class VllmStreamTransformer {
  name = "vllm-stream";

  createLineTransform() {
    const toolChoices = new Set();
    return (line) => sanitizeDataLine(line, toolChoices);
  }

  async transformResponseOut(response) {
    const contentType = response.headers.get("content-type") || "";
    if (!contentType.includes("stream") || !response.body) {
      return response;
    }

    const decoder = new TextDecoder();
    const encoder = new TextEncoder();
    const transformLine = this.createLineTransform();
    const body = new ReadableStream({
      async start(controller) {
        const reader = response.body.getReader();
        let buffered = "";
        try {
          for (;;) {
            const {done, value} = await reader.read();
            if (done) {
              break;
            }
            buffered += decoder.decode(value, {stream: true});
            let newline;
            while ((newline = buffered.indexOf("\n")) >= 0) {
              const rawLine = buffered.slice(0, newline);
              buffered = buffered.slice(newline + 1);
              const cleanLine = transformLine(rawLine);
              if (cleanLine) {
                controller.enqueue(encoder.encode(`${cleanLine}\n`));
              }
            }
          }
          buffered += decoder.decode();
          if (buffered) {
            const cleanLine = transformLine(buffered);
            if (cleanLine) {
              controller.enqueue(encoder.encode(cleanLine));
            }
          }
          controller.close();
        } catch (error) {
          controller.error(error);
        } finally {
          reader.releaseLock();
        }
      },
    });

    return new Response(body, {
      status: response.status,
      statusText: response.statusText,
      headers: response.headers,
    });
  }
}

VllmStreamTransformer.sanitizeDataLine = sanitizeDataLine;
module.exports = VllmStreamTransformer;
