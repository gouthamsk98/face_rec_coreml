import express from "express";
import { fileURLToPath } from "url";
import path from "path";
import {
  LlamaModel,
  LlamaContext,
  LlamaChatSession,
  LlamaJsonSchemaGrammar,
} from "node-llama-cpp";

const app = express();
const port = 80;

// Middleware to parse JSON bodies
app.use(express.json());

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const model = new LlamaModel({
  modelPath: path.join(
    __dirname,
    "models",
    "mistral-embedded-c-instruct-v0.4-f16.gguf"
  ),
  gpuLayers: 64,
});

const grammarSubmission = new LlamaJsonSchemaGrammar({
  type: "object",
  properties: {
    isCorrect: {
      type: "boolean",
    },
    Score: {
      type: "number",
    },
  },
});
const grammarHint = new LlamaJsonSchemaGrammar({
  type: "object",
  properties: {
    hintMessage: {
      type: "string",
    },
    hintLineNumber: {
      type: "number",
    },
  },
});
//  const context = new LlamaContext({ model });
//  const session = new LlamaChatSession({ context });
app.get("/", (req, res) => {
  res.send("alive");
});
app.post("/processUserCode", async (req, res) => {
  const context = new LlamaContext({ model });
  const session = new LlamaChatSession({ context });
  try {
    const { userCode, question, queryType } = req.body;
    let { hint_level } = req.body;
    hint_level = hint_level ? hint_level : 1;

    if (!(queryType == "hint" || queryType == "submission")) {
      res.status(400).json({ error: "queryType is wrong" });
      return;
    }
    const hintAction =
      hint_level > 2
        ? "high specificity."
        : hint_level > 1
        ? "medium specificity."
        : "simple less specificity.";
    const action =
      queryType == "submission"
        ? " is the user inputed code correct and score the answer out of 100"
        : "give me hints for the user inputed code with " + hintAction;
    const q1 =
      "For a " +
      question +
      action +
      `//user code starts
${userCode}
//user code ends`;
    console.log("USER: " + q1);
    const a1 = await session.prompt(q1, {
      grammar: queryType == "hint" ? grammarHint : grammarSubmission,
      maxTokens: context.getContextSize(),
    });
    console.log("AI: " + a1);
    res.status(200).json(a1);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Internal Server Error" });
  }
});

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
