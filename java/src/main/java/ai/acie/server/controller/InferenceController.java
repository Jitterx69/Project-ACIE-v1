package ai.acie.server.controller;

import ai.acie.server.model.InferenceRequest;
import ai.acie.server.model.InferenceResponse;
import ai.acie.server.service.CounterfactualInferenceService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.concurrent.CompletableFuture;

/**
 * REST Controller for counterfactual inference
 */
@Slf4j
@RestController
@RequestMapping("/api/v1/inference")
@RequiredArgsConstructor
public class InferenceController {

    private final CounterfactualInferenceService inferenceService;

    /**
     * Perform counterfactual inference
     * 
     * @param request Inference request with observations and interventions
     * @return Counterfactual predictions
     */
    @PostMapping("/counterfactual")
    public ResponseEntity<InferenceResponse> inferCounterfactual(
            @RequestBody InferenceRequest request) {

        log.info("Received inference request for {} observations",
                request.getObservations().length);

        InferenceResponse response = inferenceService.performInference(request);

        log.info("Inference complete, generated {} counterfactuals",
                response.getCounterfactuals().length);

        return ResponseEntity.ok(response);
    }

    /**
     * Async batch inference
     */
    @PostMapping("/counterfactual/batch")
    public CompletableFuture<ResponseEntity<InferenceResponse[]>> batchInference(
            @RequestBody InferenceRequest[] requests) {

        log.info("Received batch inference request with {} items", requests.length);

        return inferenceService.performBatchInference(requests)
                .thenApply(ResponseEntity::ok);
    }

    /**
     * Health check endpoint
     */
    @GetMapping("/health")
    public ResponseEntity<String> health() {
        return ResponseEntity.ok("ACIE Inference Server is running");
    }
}
