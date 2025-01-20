const fs = require('fs');

const { OllamaEmbeddings } = require('@langchain/ollama');
const { ChatGroq } = require('@langchain/groq');

const config = require('../config.json');
var db = require('./db/db.json');

const llm = new ChatGroq({
	model: 'llama-3.3-70b-versatile',
	temperature: 0.8,
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

const { pull } = require('langchain/hub');

const { RecursiveCharacterTextSplitter } = require('@langchain/textsplitters');
const { WebPDFLoader } = require('@langchain/community/document_loaders/web/pdf');

async function main() {
	await vectorStore.addDocuments(db);
	console.log(
		await generateResponse(
			'公認権は強いですか？参考にした資料のURLも同時に教えて下さい。'
		)
	);
}

main().catch(console.error);

async function scrape(url) {
	let docs = '';
	if (
		(await globalThis.fetch(url)).headers
			.get('content-type')
			.includes('application/pdf')
	) {
		const loader = new WebPDFLoader(url);
		docs = await loader.load();
	} else if (
		(await globalThis.fetch(url)).headers.get('content-type').includes('text/html')
	) {
		const loader = new CheerioWebBaseLoader(url);
		docs = await loader.load();
	}
	saveDocument(docs);
}

async function split(docs) {
	const splitter = new RecursiveCharacterTextSplitter({
		chunkSize: 1000,
		chunkOverlap: 200,
	});
	const allSplits = await splitter.splitDocuments(docs);
	return allSplits;
}

async function splitSentences(docs) {
	const splitter = new RecursiveCharacterTextSplitter({
		chunkSize: 1000,
		chunkOverlap: 200,
	});
	const allSplits = await splitter.createDocuments([docs]);
	return allSplits;
}

async function saveDocument(docs) {
	const allSplits = await split(docs);
	await vectorStore.addDocuments(allSplits);

	const id = fs.readFileSync('db/id').toString();
	allSplits.forEach((doc) => {
		doc.metadata.id = id;
	});
	db = [...db, ...allSplits];
	fs.writeFileSync('db/db.json', JSON.stringify(db, null, 2));
	db = JSON.parse(fs.readFileSync('db/db.json'));
	fs.writeFileSync('db/id', (parseInt(id) + 1).toString());
}

async function saveRawDocument(docs) {
	const allSplits = await splitSentences(docs);
	await vectorStore.addDocuments(allSplits);

	const id = fs.readFileSync('db/id').toString();
	allSplits.forEach((doc) => {
		doc.metadata.id = id;
	});
	db = [...db, ...allSplits];
	fs.writeFileSync('db/db.json', JSON.stringify(db, null, 2));
	db = JSON.parse(fs.readFileSync('db/db.json'));
	fs.writeFileSync('db/id', (parseInt(id) + 1).toString());
}

async function generateResponse(question) {
	const promptTemplate = await pull('rlm/rag-prompt');
	const customPrompt = `
		You are a helpful assistant. Use the following context to answer the question.
		Follow context strictly. Do not add any additional information.
		Do not answer concisely. Explain the answer in detail.
		Think step-by-step to provide a comprehensive answer.
		Context: {context}
		Question: {question}
		Answer:
	`;
	promptTemplate.lc_kwargs.promptMessages[0].prompt.template = customPrompt;
	const state = { question: question };

	const retrievedDocs = await vectorStore.similaritySearch(state.question);
	state.context = retrievedDocs.slice(0, 5);
	console.log(state.context);

	const docsContent = state.context.map((doc) => doc.pageContent).join('\n');
	const messages = await promptTemplate.invoke({
		question: state.question,
		context: docsContent,
	});
	const response = await llm.invoke(messages);

	return response.content;
}

/*
function importData(text) {
	const regex = /(https?:\/\/[^\s]+)/g; // Regex to match URLs
	const results = [];
	let match;

	while ((match = regex.exec(text)) !== null) {
		results.push(match[0]); // Add the URL to the results array
	}

	return results; // Return the array of URLs
}
*/

function importData(text) {
	const regex = /{e\d+}([\s\S]*?){e\d+}/g; // Regex to match content between {exxxx} tags
	const results = [];
	let match;

	while ((match = regex.exec(text)) !== null) {
		results.push({ content: match[1].trim() }); // Extract and trim content
	}

	return results; // Return the array of results
}
