package ai.acie.server.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Inference response model
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class InferenceResponse {

    /**
     * Request ID from original request
     */
    private String requestId;

    /**
     * Counterfactual predictions [batch_size, obs_dim]
     */
    private float[][] counterfactuals;

    /**
     * Model version used
     */
    private String modelVersion;

    /**
     * Inference time in milliseconds
     */
    private long inferenceTimeMs;

    /**
     * Error message if inference failed
     */
    private String error;
}
