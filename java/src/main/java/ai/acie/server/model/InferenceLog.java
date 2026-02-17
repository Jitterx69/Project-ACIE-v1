package ai.acie.server.model;

import jakarta.persistence.*;
import java.time.LocalDateTime;

@Entity
@Table(name = "inference_logs")
public class InferenceLog {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String requestId;
    private LocalDateTime timestamp;
    private String modelVersion;
    private Long durationMs;
    private String status;

    @Lob
    private String inputHash; // Storing hash instead of full input for privacy

    public InferenceLog() {
    }

    public InferenceLog(String requestId, String modelVersion, Long durationMs, String status, String inputHash) {
        this.requestId = requestId;
        this.timestamp = LocalDateTime.now();
        this.modelVersion = modelVersion;
        this.durationMs = durationMs;
        this.status = status;
        this.inputHash = inputHash;
    }

    // Getters and Setters
    public Long getId() {
        return id;
    }

    public String getRequestId() {
        return requestId;
    }

    public LocalDateTime getTimestamp() {
        return timestamp;
    }

    public String getModelVersion() {
        return modelVersion;
    }

    public Long getDurationMs() {
        return durationMs;
    }

    public String getStatus() {
        return status;
    }

    public String getInputHash() {
        return inputHash;
    }
}
