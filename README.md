# mirage-langchain

[![NPM](https://img.shields.io/npm/v/mirage-langchain.svg)](https://www.npmjs.com/package/mirage-langchain) [![Downloads](https://img.shields.io/npm/dt/mirage-langchain.svg)](https://www.npmjs.com/package/mirage-langchain)

The Mirage LangChain adapter. Use Mirage AI models with LangChain.

Copyright 2025 Crisp IM SAS. See LICENSE for copying information.

* **📝 Implements**: [API Reference (V1)](https://docs.mirage-ai.com/references/api/v1/) at revision: 07/04/2026
* **😘 Maintainer**: [@baptistejamin](https://github.com/baptistejamin)

## Usage

Install the library:

```bash
npm install mirage-langchain --save
```

Then, import it:

```javascript
import { ChatMirage } from "mirage-langchain";
```

Construct a new authenticated ChatMirage client with your `user_id` and `secret_key` tokens:

```javascript
const model = new ChatMirage({
  userId: "ui_xxxxxx",
  secretKey: "sk_xxxxxx",
  model: "medium"
});
```

Then, use it with LangChain:

```javascript
import { HumanMessage } from "@langchain/core/messages";

const response = await model.invoke([
  new HumanMessage("What is the capital of France?")
]);

console.log(response.content);
```

## Authentication

To authenticate against the API, get your tokens (`user_id` and `secret_key`).

Then, pass those tokens when you instantiate the ChatMirage client:

```javascript
const model = new ChatMirage({
  userId: "user_id",
  secretKey: "secret_key"
});
```

## Configuration Options

The `ChatMirage` constructor accepts the following options:

| Option | Type | Description |
|--------|------|-------------|
| `userId` | `string` | **Required.** Your Mirage API user ID |
| `secretKey` | `string` | **Required.** Your Mirage API secret key |
| `model` | `"small" \| "medium" \| "large"` | Model size to use (default: `"medium"`) |
| `temperature` | `number` | Sampling temperature |
| `maxTokens` | `number` | Maximum tokens to generate |
| `logprobs` | `boolean` | Whether to return log probabilities |
| `topLogprobs` | `number` | Number of top log probabilities to return |

## Features

### Streaming

ChatMirage supports streaming responses:

```javascript
const stream = await model.stream([
  new HumanMessage("Tell me a story")
]);

for await (const chunk of stream) {
  process.stdout.write(chunk.content);
}
```

### Tool Calling

ChatMirage supports tool calling:

```javascript
import { tool } from "@langchain/core/tools";
import { z } from "zod";

const weatherTool = tool(
  async ({ location }) => {
    return `The weather in ${location} is sunny.`;
  },
  {
    name: "get_weather",
    description: "Get the weather for a location",
    schema: z.object({
      location: z.string().describe("The city name")
    })
  }
);

const modelWithTools = model.bindTools([weatherTool]);

const response = await modelWithTools.invoke([
  new HumanMessage("What's the weather in Paris?")
]);
```

### Structured Output

ChatMirage supports structured output with Zod schemas:

```javascript
import { z } from "zod";

const schema = z.object({
  name: z.string(),
  age: z.number()
});

const structuredModel = model.withStructuredOutput(schema);

const result = await structuredModel.invoke([
  new HumanMessage("John is 25 years old")
]);

console.log(result); // { name: "John", age: 25 }
```

## Peer Dependencies

This package requires `@langchain/core` version 1.0.0 or higher as a peer dependency:

```bash
npm install @langchain/core
```
