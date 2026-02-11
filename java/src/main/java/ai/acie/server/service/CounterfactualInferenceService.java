package ai.acie.server.service;

import ai.acie.server.model.InferenceRequest;
import ai.acie.server.model.InferenceResponse;
import ai.acie.server.python.PythonModelBridge;
import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Timer;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;

import java.util.Arrays;
import java.util.concurrent.CompletableFuture;

/**
 * Core service for counterfactual inference
 * 
 * Manages:
 * - Model loading and caching
 * - Inference execution
 * - Batch processing
 * - Metrics collection
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class CounterfactualInferenceService {

    private final PythonModelBridge pythonBridge;
    private final MeterRegistry meterRegistry;

    private final Counter inferenceCounter;
    private final Timer inferenceTimer;

    public CounterfactualInferenceService(
            PythonModelBridge pythonBridge,
            MeterRegistry meterRegistry) {
        this.pythonBridge = pythonBridge;
        this.meterRegistry = meterRegistry;

        // Initialize metrics
        this.inferenceCounter = Counter.builder("acie.inference.requests")
                .description("Total inference requests")
                .register(meterRegistry);

        this.inferenceTimer = Timer.builder("acie.inference.duration")
                .description("Inference duration")
                .register(meterRegistry);
    }

    /**
     * Perform single inference
     */
    public InferenceResponse performInference(InferenceRequest request) {
        inferenceCounter.increment();

        return inferenceTimer.record(() -> {
            try {
                // Call Python model via bridge
                float[][] counterfactuals = pythonBridge.inferCounterfactual(
                        request.getObservations(),
                        request.getInterventions());

                return InferenceResponse.builder()
                        .counterfactuals(counterfactuals)
                        .requestId(request.getRequestId())
                        .modelVersion(pythonBridge.getModelVersion())
                        .build();

            } catch (Exception e) {
                log.error("Inference failed", e);
                throw new RuntimeException("Inference failed: " + e.getMessage(), e);
            }
        });
    }

    /**
     * Async batch inference
     */
    @Async("inferenceExecutor")
    public CompletableFuture<InferenceResponse[]> performBatchInference(
            InferenceRequest[] requests) {

        log.info("Processing batch of {} requests", requests.length);

        return CompletableFuture.supplyAsync(() -> {
            return Arrays.stream(requests)
                    .parallel()
                    .map(this::performInference)
                    .toArray(InferenceResponse[]::new);
        });
    }
}
