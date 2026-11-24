package coderarmy.in.avatarDemo;

import jakarta.annotation.PostConstruct;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.vectorstore.QuestionAnswerAdvisor;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.SimpleVectorStore;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.core.io.Resource;
import org.springframework.core.io.support.PathMatchingResourcePatternResolver;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Flux;

import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Service
public class AvatarService {

    private static final String SYSTEM_PROMPT = """            
            You are Friday AI, personal AI assistant of Jagrat Bisht.
            Since you are female, use female pronouns and expressions for yourself.
            
            Your job is to answer questions about Jagrat Bisht, his technical skills, projects, education, achievements, and background using retrieved knowledge.
            
            Speak in friendly, natural, and conversational Hinglish using Roman script (e.g. "Haan ji, Jagrat abhi ABES Engineering College mein 1st year B.Tech CSE Data Science ka student hai..."). Keep technical terms like TypeScript, Next.js, Node.js, C++, Open Source, AI Engineering, etc. in English.
            
            Keep answers concise, warm, conversational, and easy to understand. Prefer around 1–2 spoken sentences unless the user explicitly asks for a detailed explanation.
            
            Use retrieved knowledge as the source of truth for anything related to Jagrat Bisht.
            
            If the retrieved knowledge does not contain something specifically about Jagrat, do not invent it. You may politely mention that you don't have that specific detail, or provide a general technical explanation if asked a technical question.
                                      
            VERY IMPORTANT : WHEN USER SAYS "BYE", "GOODBYE", "OKAY BYE", "BYE FRIDAY", or similar farewell: 
            "Bye! Jagrat ke baare me baat karke bohot achha laga. Have a wonderful day!"
            """;

    private final VectorStore vectorStore;
    private final ChatClient chatClient;

    public AvatarService(ChatClient.Builder chatClientBuilder,
                         EmbeddingModel embeddingModel) {

        this.vectorStore = SimpleVectorStore.builder(embeddingModel).build();

        QuestionAnswerAdvisor ragAdvisor = QuestionAnswerAdvisor
                        .builder(vectorStore)
                        .searchRequest(SearchRequest.builder().topK(4).build())
                        .build();

        this.chatClient = chatClientBuilder
                .defaultSystem(SYSTEM_PROMPT)
                .defaultAdvisors(ragAdvisor)
                .build();
    }

    @PostConstruct
    public void loadKnowledgeBase() throws Exception {

        //1. Load the files
        PathMatchingResourcePatternResolver resolver = new PathMatchingResourcePatternResolver();
        Resource[] resources = resolver.getResources("classpath:/knowledge/*.md");

        //2. Create documents (Loop through resources and create Document)
        List<Document> documents = new ArrayList<>();

        for (Resource resource : resources) {
            String content;

            try (InputStream inputStream = resource.getInputStream()) {
                content = new String(inputStream.readAllBytes(), StandardCharsets.UTF_8);
            }

            Map<String, Object> metadata = new HashMap<>();
            metadata.put("source", resource.getFilename());

            Document document = new Document(content, metadata);
            documents.add(document);
        }

        //3. Chunking the documents (Split documents into chunks)
        TokenTextSplitter splitter = TokenTextSplitter.builder()
                .withChunkSize(400)
                .withMinChunkSizeChars(100)
                .build();

        List<Document> chunks = splitter.apply(documents);

        //4. Convert chunks into vectors and store them in a vector store
        vectorStore.write(chunks);

        System.out.println("Loaded " + chunks.size() + " knowledge chunks.");
    }

    public String ask(String question) {

        return chatClient.prompt()
                .user(question)
                .call()
                .content();
    }

    public Flux<String> askStream(String question) {

        return chatClient
                .prompt()
                .user(question)
                .stream()
                .content();
    }
}
