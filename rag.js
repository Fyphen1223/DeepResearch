const { OllamaEmbeddings } = require('@langchain/ollama');
const { ChatGroq } = require('@langchain/groq');

const config = require('./config.json');

const llm = new ChatGroq({
	model: 'llama-3.3-70b-versatile',
	temperature: 0,
	apiKey: config.groq,
});

const embeddings = new OllamaEmbeddings({
	model: 'nomic-embed-text',
});

const { MemoryVectorStore } = require('langchain/vectorstores/memory');
const vectorStore = new MemoryVectorStore(embeddings);

const {
	CheerioWebBaseLoader,
} = require('@langchain/community/document_loaders/web/cheerio');

const { ChatPromptTemplate } = require('@langchain/core/prompts');

const { pull } = require('langchain/hub');

const { Annotation, StateGraph } = require('@langchain/langgraph');

const { RecursiveCharacterTextSplitter } = require('@langchain/textsplitters');
const readline = require('readline');

async function main() {
	const pTagSelector = 'p';
	const cheerioLoader = new CheerioWebBaseLoader(
		'https://gamewith.jp/genshin/article/show/246939',
		{
			selector: pTagSelector,
		}
	);

	const docs = await cheerioLoader.load();

	const splitter = new RecursiveCharacterTextSplitter({
		chunkSize: 1000,
		chunkOverlap: 200,
	});
	const allSplits = await splitter.splitDocuments(docs);

	// Index chunks
	await vectorStore.addDocuments(allSplits);

	// Define prompt for question-answering
	const promptTemplate = await pull('rlm/rag-prompt');

	// Define state for application
	const InputStateAnnotation = Annotation.Root({
		question: Annotation,
	});

	const StateAnnotation = Annotation.Root({
		question: Annotation,
		context: Annotation,
		answer: Annotation,
	});

	// Define application steps
	const retrieve = async (state) => {
		const retrievedDocs = await vectorStore.similaritySearch(state.question);
		return { context: retrievedDocs };
	};

	const generate = async (state) => {
		const docsContent = state.context.map((doc) => doc.pageContent).join('\n');
		const messages = await promptTemplate.invoke({
			question: state.question,
			context: docsContent,
		});
		const response = await llm.invoke(messages);
		return { answer: response.content };
	};

	// Compile application and test
	const graph = new StateGraph(StateAnnotation)
		.addNode('retrieve', retrieve)
		.addNode('generate', generate)
		.addEdge('__start__', 'retrieve')
		.addEdge('retrieve', 'generate')
		.addEdge('generate', '__end__')
		.compile();

	const rl = readline.createInterface({
		input: process.stdin,
		output: process.stdout,
	});

	const askQuestion = () => {
		rl.question('質問を入力してください: ', async (input) => {
			if (input.toLowerCase() === 'exit') {
				rl.close();
				return;
			}
			let inputs = { question: input };
			const result = await graph.invoke(inputs);
			console.log(result.answer);
			askQuestion();
		});
	};

	askQuestion();
}

main().catch(console.error);
