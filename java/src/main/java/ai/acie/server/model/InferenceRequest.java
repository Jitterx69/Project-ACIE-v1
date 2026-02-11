package ai.acie.server.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.Map;

/**
 * Inference request model
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class InferenceRequest {

    /**
     * Unique request identifier
     */
    private String requestId;

    /**
     * Input observations [batch_size, obs_dim]
     */
    private float[][] observations;

    /**
     * Interventions to apply
     * Key: variable name (e.g., "mass", "metallicity")
     * Value: intervention value
     */
    private Map<String, Float> interventions;

    /**
     * Model version to use (optional)
     */
    private String modelVersion;
}
