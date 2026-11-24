package coderarmy.in.avatarDemo;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.AbstractEmbeddingModel;
import org.springframework.ai.embedding.Embedding;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.embedding.EmbeddingRequest;
import org.springframework.ai.embedding.EmbeddingResponse;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Primary;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Configuration
public class GeminiEmbeddingConfig {

    @Value("${spring.ai.openai.api-key}")
    private String apiKey;

    @Value("${spring.ai.openai.embedding.model:gemini-embedding-001}")
    private String embeddingModelName;

    private final ObjectMapper objectMapper = new ObjectMapper();

    @Bean
    @Primary
    public EmbeddingModel geminiEmbeddingModel() {
        return new AbstractEmbeddingModel() {
            private final HttpClient client = HttpClient.newBuilder()
                    .connectTimeout(Duration.ofSeconds(30))
                    .build();

            @Override
            public EmbeddingResponse call(EmbeddingRequest request) {
                List<String> instructions = request.getInstructions();
                List<Embedding> embeddings = new ArrayList<>();

                for (int i = 0; i < instructions.size(); i++) {
                    String text = instructions.get(i);
                    float[] vector = fetchEmbedding(text);
                    embeddings.add(new Embedding(vector, i));
                }

                return new EmbeddingResponse(embeddings);
            }

            @Override
            public float[] embed(Document document) {
                return fetchEmbedding(document.getText());
            }

            @Override
            public int dimensions() {
                return 3072;
            }

            private float[] fetchEmbedding(String text) {
                try {
                    Map<String, Object> payload = new HashMap<>();
                    payload.put("model", embeddingModelName);
                    payload.put("input", text);

                    String json = objectMapper.writeValueAsString(payload);

                    HttpRequest httpRequest = HttpRequest.newBuilder()
                            .uri(URI.create("https://generativelanguage.googleapis.com/v1beta/openai/embeddings"))
                            .header("Authorization", "Bearer " + apiKey)
                            .header("Content-Type", "application/json")
                            .timeout(Duration.ofSeconds(30))
                            .POST(HttpRequest.BodyPublishers.ofString(json))
                            .build();

                    HttpResponse<String> response = client.send(httpRequest, HttpResponse.BodyHandlers.ofString());
                    if (response.statusCode() != 200) {
                        throw new RuntimeException("Gemini embedding failed with status " + response.statusCode() + ": " + response.body());
                    }

                    JsonNode root = objectMapper.readTree(response.body());
                    JsonNode data = root.path("data");
                    if (!data.isArray() || data.isEmpty()) {
                        throw new RuntimeException("No embedding returned: " + response.body());
                    }

                    JsonNode embeddingNode = data.get(0).path("embedding");
                    float[] vector = new float[embeddingNode.size()];
                    for (int j = 0; j < embeddingNode.size(); j++) {
                        vector[j] = (float) embeddingNode.get(j).asDouble();
                    }
                    return vector;
                } catch (Exception e) {
                    throw new RuntimeException("Failed to embed text with Gemini: " + e.getMessage(), e);
                }
            }
        };
    }
}
