package ai.acie.server.service;

import ai.acie.server.model.InferenceLog;
import ai.acie.server.repository.InferenceLogRepository;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;

import java.util.concurrent.CompletableFuture;

@Service
public class AuditService {

    private final InferenceLogRepository logRepository;

    public AuditService(InferenceLogRepository logRepository) {
        this.logRepository = logRepository;
    }

    @Async
    public CompletableFuture<Void> logInference(String requestId, String modelVersion, long duration, String status,
            String inputHash) {
        InferenceLog log = new InferenceLog(requestId, modelVersion, duration, status, inputHash);
        logRepository.save(log);
        return CompletableFuture.completedFuture(null);
    }
}
